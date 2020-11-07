#include "./kernel_spmspv.h"

#include <iostream>
#include <iomanip>

#include <ap_fixed.h>
#include <hls_stream.h>

#include "./shuffle.h"
#include "./pe.h"

#ifndef __SYNTHESIS__
static bool line_tracing_spmspv = true;
#endif


// vector loader for spmspv
static void load_vector_from_gmem(
    // vector data, row_id
    VEC_PKT_T *vector,
    // number of non-zeros
    IDX_T vec_num_nnz,
    // fifo
    hls::stream<VL_O_T> &VL_to_ML_stream
) {
    loop_over_vector_values:
    for (unsigned int vec_nnz_cnt = 0; vec_nnz_cnt < vec_num_nnz; vec_nnz_cnt++) {
        #pragma HLS pipeline II=1
        VL_O_T instruction_to_ml;
        instruction_to_ml.current_column_id = vector[vec_nnz_cnt + 1].index;
        instruction_to_ml.vector_value = vector[vec_nnz_cnt + 1].val;
        VL_to_ML_stream.write(instruction_to_ml);
    }
}

// data loader for spmspv
static void load_matrix_from_gmem(
    // matrix data, row_id
    MAT_PKT_T *matrix,
    // matrix indptr
    IDX_T *mat_indptr,
    // matrix part ptr
    IDX_T *mat_partptr,
    // number of non-zeros
    IDX_T vec_num_nnz,
    // partition base
    IDX_T mat_indptr_base,
    IDX_T mat_row_id_base,
    // current part id
    IDX_T part_id,
    VAL_T Zero,
    // fifos
    hls::stream<VL_O_T> &VL_to_ML_stream,
    hls::stream<SF_IO_T> DL_to_SF_stream[PACK_SIZE],
    // load complete
    hls::stream<unsigned> &num_payloads
) {
    IDX_T pld_cnt = 0;
    IDX_T mat_addr_base = mat_partptr[part_id];
    // loop over all active columns
    loop_over_active_columns_ML:
    for (unsigned int vec_nnz_cnt = 0; vec_nnz_cnt < vec_num_nnz; vec_nnz_cnt++) {

        // slice out the current column out of the active columns
        VL_O_T instruction_from_vl;
        VL_to_ML_stream.read(instruction_from_vl);
        IDX_T current_colid = instruction_from_vl.current_column_id;
        VAL_T vec_val = instruction_from_vl.vector_value;

        // [0] for start, [1] for end
        // write like this to make sure it uses burst read
        IDX_T col_slice[2];
        #pragma HLS array_partition variable=col_slice complete

        loop_get_column_len_ML:
        for (unsigned int i = 0; i < 2; i++) {
            #pragma HLS pipeline II=1
            col_slice[i] = mat_indptr[current_colid + mat_indptr_base + i];
        }

        IDX_T tmp_pcnt = 0;
        loop_over_pkts_ML:
        for (unsigned int i = 0; i < (col_slice[1] - col_slice[0]); i++) {
            #pragma HLS pipeline II=1

            MAT_PKT_T packet_from_mat = matrix[i + mat_addr_base + col_slice[0]];

            IDX_T tmp_pcnt_incr = 0;
            loop_unpack_ML_unroll:
            for (unsigned int k = 0; k < PACK_SIZE; k++) {
                #pragma HLS unroll

                SF_IO_T input_to_SF;
                input_to_SF.data.mat_val = packet_from_mat.vals.data[k];
                input_to_SF.data.vec_val = vec_val;
                input_to_SF.index = packet_from_mat.indices.data[k] - mat_row_id_base;
                // discard paddings
                if (packet_from_mat.vals.data[k] != Zero) {
                    DL_to_SF_stream[k].write(input_to_SF);
                    tmp_pcnt_incr ++;
                }
            }
            tmp_pcnt += tmp_pcnt_incr;
        }
        pld_cnt += tmp_pcnt;
    }
    num_payloads.write(pld_cnt);

}

// bram access used for checkout results
void bram_access_read_2ports(
    // real read ports
    IDX_T rd_addr[PACK_SIZE * 2],
    VAL_T rd_data[PACK_SIZE * 2],
    // bram
    VAL_T bram[PACK_SIZE][OUT_BUF_LEN / PACK_SIZE]
){
    #pragma HLS pipeline II=1
    // #pragma HLS inline

    loop_rd_get_data_unroll:
    for (unsigned int BKid = 0; BKid < PACK_SIZE; BKid++) {
        #pragma HLS unroll
        for (unsigned int PTid = 0; PTid < 2; PTid++) {
            #pragma HLS unroll
            IDX_T sbbk_addr = rd_addr[BKid * 2 + PTid] >> BANK_ID_NBITS;
            rd_data[BKid * 2 + PTid] = bram[BKid][sbbk_addr];
        }
    }
}

// change results to sparse
static void checkout_results(
    // data to be checked
    VAL_T dense_data[PACK_SIZE][OUT_BUF_LEN / PACK_SIZE],
    // FIFOs
    hls::stream<VEC_PKT_T> cr_output_streams[PACK_SIZE * 2],
    // control signals
    hls::stream<IDX_T> &npld_to_wb,
    IDX_T mat_row_id_base,
    IDX_T num_rows,
    VAL_T zero
) {
    IDX_T npld_before_mask = 0;
    IDX_T index_arr[PACK_SIZE * 2];
    VAL_T data_arr[PACK_SIZE * 2];
    #pragma HLS array_partition variable=index_arr complete
    #pragma HLS array_partition variable=data_arr complete

    unsigned int num_rounds = (num_rows + 2 * PACK_SIZE - 1) / (PACK_SIZE * 2);
    loop_over_dense_data_pipeline:
    for (unsigned int round_cnt = 0; round_cnt < num_rounds; round_cnt ++) {
        #pragma HLS pipeline II=1
        IDX_T local_npld_incr = 0;

        loop_before_read_banks_unroll:
        for (unsigned int Bank_id = 0; Bank_id < PACK_SIZE; Bank_id++) {
            #pragma HLS unroll
            loop_before_read_ports_unroll:
            for (unsigned int Port_id = 0; Port_id < 2; Port_id++) {
                #pragma HLS unroll
                index_arr[Bank_id * 2 + Port_id] = Bank_id + (round_cnt * 2 + Port_id) * PACK_SIZE;
            }
        }

        bram_access_read_2ports(index_arr,data_arr,dense_data);

        loop_after_read_banks_unroll:
        for (unsigned int Bank_id = 0; Bank_id < PACK_SIZE; Bank_id++) {
            #pragma HLS unroll
            loop_after_read_ports_unroll:
            for (unsigned int Port_id = 0; Port_id < 2; Port_id++) {
                #pragma HLS unroll
                if (data_arr[Bank_id * 2 + Port_id] != zero) {
                    VEC_PKT_T pld;
                    pld.index = index_arr[Bank_id * 2 + Port_id] + mat_row_id_base;
                    pld.val = data_arr[Bank_id * 2 + Port_id];
                    cr_output_streams[Bank_id * 2 + Port_id].write(pld);
                    local_npld_incr ++;
                }
            }
        }
        npld_before_mask += local_npld_incr;
    }
    npld_to_wb << npld_before_mask;
}


// write back to ddr (out-of-order)
static void write_back_gmem(
    hls::stream<VEC_PKT_T> wb_input_streams[PACK_SIZE * 2],
    hls::stream<IDX_T> &npld_stream,
    VAL_T *mask,
    VEC_PKT_T *result,
    IDX_T Nnz,
    IDX_T &Nnz_incr,
    VAL_T zero,
    MASK_T mask_type
) {
    IDX_T wb_cnt = 0;
    IDX_T incr = 0;
    IDX_T npld = 0;
    bool checkout_finish = false;
    bool loop_exit = false;
    unsigned int Lane_id = 0;

    loop_until_all_written_back:
    while (!loop_exit) {
        #pragma HLS pipeline II=1
        #pragma HLS dependence variable=loop_exit inter distance=15 RAW True
        VEC_PKT_T wb_temp;
        if (wb_input_streams[Lane_id].read_nb(wb_temp)) {
            wb_cnt ++;
            bool do_write;
            switch (mask_type) {
                case NOMASK:
                    do_write = true;
                    break;
                case WRITETOONE:
                    do_write = (mask[wb_temp.index] != zero);
                    break;
                case WRITETOZERO:
                    do_write = (mask[wb_temp.index] == zero);
                    break;
                default:
                    do_write = false;
                    break;
            }
            if (do_write) {
                result[Nnz + incr + 1] = wb_temp;
                incr ++;
            }
        }
        if (!checkout_finish) {
            checkout_finish = npld_stream.read_nb(npld);
        }
        Lane_id = (Lane_id + 1) % (PACK_SIZE * 2);
        loop_exit = checkout_finish && (npld == wb_cnt);
    }
    Nnz_incr = incr;
}

// reset output buffer
static void reset_output_buffer(
    VAL_T output_buffer[PACK_SIZE][OUT_BUF_LEN / PACK_SIZE],
    IDX_T num_rows,
    VAL_T Zero
) {
    #pragma HLS pipeline II=1
    unsigned int num_rounds = (num_rows + 2 * PACK_SIZE - 1) / PACK_SIZE;
    loop_reset_ob:
    for (unsigned int i = 0; i < num_rounds; i++) {
        #pragma HLS pipeline II=1
        for (unsigned int j = 0; j < PACK_SIZE; j++) {
            #pragma HLS unroll
            output_buffer[j][i] = Zero;
        }
    }

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Output Buffer reset complete" << std::endl << std::flush;
    }
    #endif
}

// compute
static void compute_spmspv(
    MAT_PKT_T *matrix,
    IDX_T *mat_indptr,
    IDX_T *mat_partptr,
    VEC_PKT_T *vector,
    VAL_T output_buffer[PACK_SIZE][OUT_BUF_LEN / PACK_SIZE],
    IDX_T vec_num_nnz,
    IDX_T mat_indptr_base,
    IDX_T mat_row_id_base,
    IDX_T part_id,
    OP_T Op,
    VAL_T Zero
) {
    // fifos
    hls::stream<IDX_T> DL_to_SF_npld_stream;
    hls::stream<IDX_T> SF_to_PE_npld_stream;
    hls::stream<VL_O_T> VL_to_ML_stream;
    hls::stream<SF_IO_T> DL_to_SF_stream[PACK_SIZE];
    hls::stream<SF_IO_T> SF_to_PE_stream[PACK_SIZE];
    #pragma HLS stream variable=DL_to_SF_npld_stream depth=2
    #pragma HLS stream variable=SF_to_PE_npld_stream depth=2
    #pragma HLS stream variable=VL_to_ML_stream depth=64
    #pragma HLS stream variable=DL_to_SF_stream depth=64
    #pragma HLS stream variable=SF_to_PE_stream depth=64

    // dataflow pipeline
    #pragma HLS dataflow
    load_vector_from_gmem(
        vector,
        vec_num_nnz,
        VL_to_ML_stream
    );

    load_matrix_from_gmem(
        matrix,
        mat_indptr,
        mat_partptr,
        vec_num_nnz,
        mat_indptr_base,
        mat_row_id_base,
        part_id,
        Zero,
        VL_to_ML_stream,
        DL_to_SF_stream,
        DL_to_SF_npld_stream
    );

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO :[Kernel SpMSpV] Data Loader complete" << std::endl << std::flush;
    }
    #endif

    shuffler_1p<SF_IO_T, SF_IO_VAL_T, PACK_SIZE, PACK_SIZE, BANK_ID_MASK>(
        DL_to_SF_stream,
        SF_to_PE_stream,
        DL_to_SF_npld_stream,
        SF_to_PE_npld_stream
    );

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO :[Kernel SpMSpV] Shuffler complete" << std::endl << std::flush;
    }
    #endif

    pe_cluster<VAL_T, OP_T, SF_IO_T, PACK_SIZE, BANK_ID_NBITS>(
        SF_to_PE_stream,
        output_buffer,
        Op,
        Zero,
        SF_to_PE_npld_stream
    );

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO :[Kernel SpMSpV] Process Elements complete" << std::endl << std::flush;
    }
    #endif


    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO :[Kernel SpMSpV] Computation complete" << std::endl << std::flush;
    }
    #endif
}

// write back
static void write_back_results(
    VAL_T output_buffer[PACK_SIZE][OUT_BUF_LEN / PACK_SIZE],
    VEC_PKT_T *result,
    VAL_T *mask,
    IDX_T num_rows,
    IDX_T result_Nnz,
    IDX_T &result_Nnz_incr,
    IDX_T mat_row_id_base,
    VAL_T zero,
    MASK_T mask_type
) {
    hls::stream<VEC_PKT_T> CR_to_WB_stream[PACK_SIZE * 2];
    hls::stream<IDX_T> CR_to_WB_npld_stream;
    #pragma HLS stream variable=CR_to_WB_npld_stream depth=2
    #pragma HLS stream variable=CR_to_WB_stream depth=64

    // dataflow pipeline
    #pragma HLS dataflow

    checkout_results(
        output_buffer,
        CR_to_WB_stream,
        CR_to_WB_npld_stream,
        mat_row_id_base,
        num_rows,
        zero
    );
    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Checkout Results complete" << std::endl << std::flush;
    }
    #endif

    write_back_gmem(
        CR_to_WB_stream,
        CR_to_WB_npld_stream,
        mask,
        result,
        result_Nnz,
        result_Nnz_incr,
        zero,
        mask_type
    );

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Result writeback complete" << std::endl << std::flush;
    }
    #endif
}

extern "C" {
void kernel_spmspv(
    MAT_PKT_T *matrix,   // 0
    IDX_T *mat_indptr,   // 1
    IDX_T *mat_partptr,  // 2
    VEC_PKT_T *vector,   // 3
    VAL_T *mask,         // 4
    VEC_PKT_T *result,   // 5
    IDX_T num_rows,      // 6
    IDX_T num_cols,      // 7
    OP_T Op,             // 8
    MASK_T Mask_type     // 9
) {
    #pragma HLS data_pack variable=matrix
    #pragma HLS data_pack variable=vector
    #pragma HLS data_pack variable=result

    #pragma HLS interface m_axi port=matrix       offset=slave bundle=gmem1
    #pragma HLS interface m_axi port=mat_indptr   offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=mat_partptr  offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=mask         offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=vector       offset=slave bundle=gmem2
    #pragma HLS interface m_axi port=result       offset=slave bundle=gmem2

    #pragma HLS interface s_axilite port=matrix       bundle=control
    #pragma HLS interface s_axilite port=mat_indptr   bundle=control
    #pragma HLS interface s_axilite port=mat_partptr  bundle=control
    #pragma HLS interface s_axilite port=mask         bundle=control
    #pragma HLS interface s_axilite port=vector       bundle=control
    #pragma HLS interface s_axilite port=result       bundle=control

    #pragma HLS interface s_axilite port=num_rows   bundle=control
    #pragma HLS interface s_axilite port=num_cols   bundle=control
    #pragma HLS interface s_axilite port=Op         bundle=control
    #pragma HLS interface s_axilite port=Mask_type  bundle=control

    #pragma HLS interface s_axilite port=return     bundle=control

    IDX_T vec_num_nnz = vector[0].index;
    VAL_T Zero;

    switch (Op) {
    case MULADD:
        Zero = MulAddZero;
        break;
    case ANDOR:
        Zero = AndOrZero;
        break;
    case ADDMIN:
        Zero = AddMinZero;
        break;
    default:
        Zero = 0;
        break;
    }

    // output buffer
    VAL_T output_buffer[PACK_SIZE][OUT_BUF_LEN / PACK_SIZE];
    #pragma HLS array_partition variable=output_buffer dim=1 complete
    #pragma HLS resource variable=output_buffer core=RAM_2P

    // result Nnz counter
    IDX_T result_Nnz = 0;

    // total number of parts
    IDX_T num_parts = (num_rows + OUT_BUF_LEN - 1) / OUT_BUF_LEN;

    // number of rows in the last part
    IDX_T num_rows_last_part = (num_rows % OUT_BUF_LEN) ? (num_rows % OUT_BUF_LEN) : OUT_BUF_LEN;

    // loop over parts
    loop_over_parts:
    for (unsigned int part_id = 0; part_id < num_parts; part_id++) {
        #pragma HLS pipeline off
        IDX_T num_rows_this_part = (part_id == (num_parts - 1)) ? num_rows_last_part : OUT_BUF_LEN;
        IDX_T mat_indptr_base = (num_cols + 1) * part_id;
        IDX_T mat_row_id_base = OUT_BUF_LEN * part_id;
        IDX_T result_Nnz_incr;
        reset_output_buffer(
            output_buffer,
            num_rows_this_part,
            Zero
        );
        compute_spmspv(
            matrix,
            mat_indptr,
            mat_partptr,
            vector,
            output_buffer,
            vec_num_nnz,
            mat_indptr_base,
            mat_row_id_base,
            part_id,
            Op,
            Zero
        );
        write_back_results(
            output_buffer,
            result,
            mask,
            num_rows_this_part,
            result_Nnz,
            result_Nnz_incr,
            mat_row_id_base,
            Zero,
            Mask_type
        );
        result_Nnz += result_Nnz_incr;
        #ifndef __SYNTHESIS__
        if (line_tracing_spmspv) {
            std::cout << "INFO: [Kernel SpMSpV] Partition " << part_id <<" complete" << std::endl << std::flush;
            std::cout << "  # of rows this part: " << num_rows_this_part << std::endl << std::flush;
            std::cout << "          row id base: " << mat_row_id_base << std::endl << std::flush;
            std::cout << "          indptr base: " << mat_indptr_base << std::endl << std::flush;
            std::cout << "     Nnz written back: " << result_Nnz << std::endl << std::flush;
        }
        #endif
    }

    // attach head
    VEC_PKT_T result_head;
    result_head.index = result_Nnz;
    result_head.val = Zero;
    result[0] = result_head;
    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Kernel Finish" << std::endl << std::flush;
        std::cout << "  Result Nnz = " << result_Nnz << std::endl << std::flush;
    }
    #endif
}

}  // extern "C"
