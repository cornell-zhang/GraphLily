#include "libfpga/hisparse.h"
#include "libfpga/shuffle.h"
#include "libfpga/pe.h"

#include <ap_fixed.h>
#include <hls_stream.h>

#ifndef __SYNTHESIS__
#include <iostream>
#include <iomanip>
static bool line_tracing_spmspv = false;
static bool line_tracing_spmspv_checkout = false;
#endif

// typedef struct shuffle_inout_vale_type {
//     VAL_T mat_val;
//     VAL_T vec_val;
// } SF_IO_VAL_T;

// typedef struct shuffle_inout_type {
//     IDX_T index;
//     SF_IO_VAL_T data;
// } SF_IO_T;

// typedef struct vector_loader_out_type {
//     IDX_T current_column_id;
//     VAL_T vector_value;
// } VL_O_T;


// vector loader for spmspv
void load_vector_from_gmem(
    // vector data, row_id
    const IDX_VAL_T *vector,
    // number of non-zeros
    IDX_T vec_num_nnz,
    // fifo
    hls::stream<IDX_VAL_T> &VL_to_ML_stream
) {
    loop_over_vector_values:
    for (unsigned int vec_nnz_cnt = 0; vec_nnz_cnt < vec_num_nnz; vec_nnz_cnt++) {
        #pragma HLS pipeline II=1
        IDX_VAL_T instruction_to_ml;
        instruction_to_ml.index = vector[vec_nnz_cnt + 1].index;
        instruction_to_ml.val = vector[vec_nnz_cnt + 1].val;
        VL_to_ML_stream.write(instruction_to_ml);
    }
}


// data loader for spmspv
void load_matrix_from_gmem(
    // matrix data, row_id
    const SPMSPV_MAT_PKT_T *matrix,
    // matrix indptr
    const IDX_T *mat_indptr,
    // matrix part ptr
    const IDX_T *mat_partptr,
    // number of non-zeros
    IDX_T vec_num_nnz,
    // partition base
    IDX_T mat_indptr_base,
    IDX_T mat_row_id_base,
    // current part id
    IDX_T part_id,
    VAL_T Zero,
    // fifos
    hls::stream<IDX_VAL_T> &VL_to_ML_stream,
    hls::stream<UPDATE_PLD_T> DL_to_SF_stream[PACK_SIZE]
) {
    IDX_T mat_addr_base = mat_partptr[part_id];

    for (unsigned int k = 0; k < PACK_SIZE; k++) {
        #pragma HLS unroll
        DL_to_SF_stream[k].write(UPDATE_PLD_SOD);
    }

    // loop over all active columns
    loop_over_active_columns_ML:
    for (unsigned int vec_nnz_cnt = 0; vec_nnz_cnt < vec_num_nnz; vec_nnz_cnt++) {

        // slice out the current column out of the active columns
        IDX_VAL_T instruction_from_vl;
        VL_to_ML_stream.read(instruction_from_vl);
        IDX_T current_colid = instruction_from_vl.index;
        VAL_T vec_val = instruction_from_vl.val;

        // [0] for start, [1] for end
        // write like this to make sure it uses burst read
        IDX_T col_slice[2];
        #pragma HLS array_partition variable=col_slice complete

        loop_get_column_len_ML:
        for (unsigned int i = 0; i < 2; i++) {
            #pragma HLS pipeline II=1
            col_slice[i] = mat_indptr[current_colid + mat_indptr_base + i];
        }

        loop_over_pkts_ML:
        for (unsigned int i = 0; i < (col_slice[1] - col_slice[0]); i++) {
            #pragma HLS pipeline II=1
            SPMSPV_MAT_PKT_T packet_from_mat = matrix[i + mat_addr_base + col_slice[0]];

            loop_unpack_ML_unroll:
            for (unsigned int k = 0; k < PACK_SIZE; k++) {
                #pragma HLS unroll
                UPDATE_PLD_T input_to_SF;
                input_to_SF.mat_val = packet_from_mat.vals.data[k];
                input_to_SF.vec_val = vec_val;
                input_to_SF.row_idx = packet_from_mat.indices.data[k] - mat_row_id_base;
                input_to_SF.inst = 0;
                // discard paddings
                if (packet_from_mat.vals.data[k] != Zero) {
                    DL_to_SF_stream[k].write(input_to_SF);
                }
            }
        }
    }

    for (unsigned int k = 0; k < PACK_SIZE; k++) {
        #pragma HLS unroll
        DL_to_SF_stream[k].write(UPDATE_PLD_EOD);
        DL_to_SF_stream[k].write(UPDATE_PLD_EOS);
    }
}


// // bram access used for checkout results
// void bram_access_read_2ports(
//     // real read ports
//     IDX_T rd_addr0[PACK_SIZE],
//     VAL_T rd_data0[PACK_SIZE],
//     IDX_T rd_addr1[PACK_SIZE],
//     VAL_T rd_data1[PACK_SIZE],
//     // bram
//     const VAL_T bram[PACK_SIZE][SPMSPV_OUT_BUF_LEN / PACK_SIZE]
// ) {
//     #pragma HLS pipeline II=1
//     // #pragma HLS inline

//     loop_rd_get_data_unroll:
//     for (unsigned int BKid = 0; BKid < PACK_SIZE; BKid++) {
//         #pragma HLS unroll
//         rd_data0[BKid] = bram[rd_addr0[BKid] % NUM_HBM_CHANNEL][BKid][rd_addr0[BKid] / NUM_HBM_CHANNEL];
//         rd_data1[BKid] = bram[rd_addr1[BKid] % NUM_HBM_CHANNEL][BKid][rd_addr1[BKid] / NUM_HBM_CHANNEL];
//     }
// }


// // change results to sparse
// void checkout_results(
//     // data to be checked
//     const VAL_T dense_data[PACK_SIZE][SPMSPV_OUT_BUF_LEN / PACK_SIZE],
//     // FIFOs
//     hls::stream<IDX_VAL_T> cr_output_streams[PACK_SIZE * 2],
//     // control signals
//     hls::stream<IDX_T> &npld_to_wb,
//     IDX_T mat_row_id_base,
//     IDX_T num_rows,
//     VAL_T zero
// ) {
//     IDX_T npld_before_mask = 0;
//     IDX_T index_arr0[PACK_SIZE];
//     IDX_T index_arr1[PACK_SIZE];
//     VAL_T data_arr0[PACK_SIZE];
//     VAL_T data_arr1[PACK_SIZE];
//     #pragma HLS array_partition variable=index_arr0 complete
//     #pragma HLS array_partition variable=index_arr1 complete
//     #pragma HLS array_partition variable=data_arr0 complete
//     #pragma HLS array_partition variable=data_arr1 complete

//     unsigned int num_rounds = ((num_rows + 2 * PACK_SIZE) - 1) / (PACK_SIZE * 2);
//     loop_over_dense_data_pipeline:
//     for (unsigned int round_cnt = 0; round_cnt < num_rounds; round_cnt++) {
//         #pragma HLS pipeline II=1
//         IDX_T local_npld_incr = 0;

//         loop_before_read_banks_unroll:
//         for (unsigned int Bank_id = 0; Bank_id < PACK_SIZE; Bank_id++) {
//             #pragma HLS unroll
//             index_arr0[Bank_id] = round_cnt * 2;
//             index_arr1[Bank_id] = round_cnt * 2 + 1;
//         }

//         bram_access_read_2ports(index_arr0, data_arr0, index_arr1, data_arr1, dense_data);

//         loop_after_read_banks_unroll:
//         for (unsigned int Bank_id = 0; Bank_id < PACK_SIZE; Bank_id++) {
//             #pragma HLS unroll
//             if (data_arr0[Bank_id] != zero) {
//                 IDX_VAL_T pld;
//                 pld.index = round_cnt * 2 * PACK_SIZE + Bank_id + mat_row_id_base;
//                 pld.val = data_arr0[Bank_id];
//                 cr_output_streams[Bank_id].write(pld);
//                 local_npld_incr++;

//             }
//             if (data_arr1[Bank_id] != zero) {
//                 IDX_VAL_T pld;
//                 pld.index = (round_cnt * 2 + 1) * PACK_SIZE + Bank_id + mat_row_id_base;
//                 pld.val = data_arr1[Bank_id];
//                 cr_output_streams[Bank_id + PACK_SIZE].write(pld);
//                 local_npld_incr++;

//             }
//         }
//         npld_before_mask += local_npld_incr;
//     }
//     npld_to_wb << npld_before_mask;
// }

// write back to gemm
static void write_back_results (
    hls::stream<VEC_PLD_T> PE_to_WB_stream[PACK_SIZE],
    IDX_VAL_T *res_out,
    const VAL_T *mask,
    IDX_T mat_row_id_base,
    IDX_T Nnz,
    IDX_T &Nnz_incr,
    VAL_T Zero,
    MASK_T mask_type
) {
    bool exit = false;
    IDX_T incr = 0;
    while (!exit) {
        #pragma HLS pipeline II=1
        ap_uint<PACK_SIZE> got_EOS = 0;
        for (unsigned k = 0; k < PACK_SIZE; k++) {
            #pragma HLS unroll
            VEC_PLD_T p = PE_to_WB_stream[k].read();
            if (p.inst == EOS) {
                got_EOS[k] = 1;
            } else if (p.inst != SOD && p.inst != EOD) {
                got_EOS[k] = 0;
                if (p.val != Zero) {
                    IDX_T index = mat_row_id_base + p.idx;
                    bool do_write = false;
                    switch (mask_type) {
                        case NOMASK:
                            do_write = true;
                            break;
                        case WRITETOONE:
                            do_write = (mask[index] != Zero);
                            break;
                        case WRITETOZERO:
                            do_write = (mask[index] == Zero);
                            break;
                        default:
                            do_write = false;
                            break;
                    }
                    if (do_write) {
                        incr++;
                        res_out[Nnz + incr] = (IDX_VAL_T){.index=index, .val=p.val};
                        #ifndef __SYNTHESIS__
                        if (line_tracing_spmspv_checkout) {
                            std::cout << "INFO: [kernel SpMSpV write results]"
                                        << " non-zero " << p.val
                                        << " found at " << p.idx
                                        << " mapped to " << index << std::endl << std::flush;
                        }
                        #endif
                    }
                }
            }
        }
        if (got_EOS.and_reduce()) {
            exit = true;
        }
    }

    Nnz_incr = incr;
}

// load vec/mat -> shuffle -> pe compute -> writeback
void spmspv_core(
    const SPMSPV_MAT_PKT_T *matrix,
    const IDX_T *mat_indptr,
    const IDX_T *mat_partptr,
    const IDX_VAL_T *vector,
    const VAL_T *mask,
    IDX_VAL_T *result_out,
    IDX_T vec_num_nnz,
    IDX_T mat_indptr_base,
    IDX_T mat_row_id_base,
    IDX_T part_id,
    IDX_T Nnz,
    IDX_T &Nnz_incr,
    OP_T Op,
    VAL_T Zero,
    MASK_T mask_type,
    const unsigned used_buf_len_per_pe
) {
    // fifos
    // hls::stream<IDX_T> DL_to_SF_npld_stream;
    // hls::stream<IDX_T> SF_to_PE_npld_stream;
    // #pragma HLS stream variable=DL_to_SF_npld_stream depth=2
    // #pragma HLS stream variable=SF_to_PE_npld_stream depth=2

    hls::stream<IDX_VAL_T> VL_to_ML_stream;
    hls::stream<UPDATE_PLD_T> DL_to_SF_stream[PACK_SIZE];
    hls::stream<UPDATE_PLD_T> SF_to_PE_stream[PACK_SIZE];
    hls::stream<VEC_PLD_T> PE_to_WB_stream[PACK_SIZE];
    #pragma HLS stream variable=VL_to_ML_stream depth=FIFO_DEPTH
    #pragma HLS stream variable=DL_to_SF_stream depth=FIFO_DEPTH
    #pragma HLS stream variable=SF_to_PE_stream depth=FIFO_DEPTH
    #pragma HLS stream variable=PE_to_WB_stream depth=FIFO_DEPTH
    #pragma HLS RESOURCE variable=VL_to_ML_stream core=FIFO_SRL
    #pragma HLS RESOURCE variable=DL_to_SF_stream core=FIFO_SRL
    #pragma HLS RESOURCE variable=SF_to_PE_stream core=FIFO_SRL
    #pragma HLS RESOURCE variable=PE_to_WB_stream core=FIFO_SRL

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
        DL_to_SF_stream
    );

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Data Loader complete" << std::endl << std::flush;
    }
    #endif

    shuffler<UPDATE_PLD_T, PACK_SIZE>(DL_to_SF_stream, SF_to_PE_stream);

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Shuffler complete" << std::endl << std::flush;
    }
    #endif

    pe_bram<0, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[0],
        PE_to_WB_stream[0],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram<1, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[1],
        PE_to_WB_stream[1],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram<2, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[2],
        PE_to_WB_stream[2],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram<3, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[3],
        PE_to_WB_stream[3],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram<4, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[4],
        PE_to_WB_stream[4],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram<5, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[5],
        PE_to_WB_stream[5],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram<6, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[6],
        PE_to_WB_stream[6],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram<7, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[7],
        PE_to_WB_stream[7],
        used_buf_len_per_pe,
        Op,
        Zero
    );

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Process Elements complete" << std::endl << std::flush;
    }
    #endif

    write_back_results(
        PE_to_WB_stream,
        result_out,
        mask,
        mat_row_id_base,
        Nnz,
        Nnz_incr,
        Zero,
        mask_type
    );

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Result writeback complete" << std::endl << std::flush;
    }
    #endif
}


// // write back
// void write_back_results(
//     const VAL_T output_buffer[PACK_SIZE][SPMSPV_OUT_BUF_LEN / PACK_SIZE],
//     IDX_VAL_T *result,
//     const VAL_T *mask,
//     IDX_T num_rows,
//     IDX_T result_Nnz,
//     IDX_T &result_Nnz_incr,
//     IDX_T mat_row_id_base,
//     VAL_T zero,
//     MASK_T mask_type
// ) {
//     hls::stream<IDX_T> CR_to_WB_npld_stream;
//     #pragma HLS stream variable=CR_to_WB_npld_stream depth=2

//     hls::stream<IDX_VAL_T> CR_to_WB_stream[PACK_SIZE * 2];
//     #pragma HLS stream variable=CR_to_WB_stream depth=FIFO_DEPTH
//     #pragma HLS RESOURCE variable=CR_to_WB_stream core=FIFO_SRL

//     // dataflow pipeline
//     #pragma HLS dataflow

//     checkout_results(
//         output_buffer,
//         CR_to_WB_stream,
//         CR_to_WB_npld_stream,
//         mat_row_id_base,
//         num_rows,
//         zero
//     );
//     #ifndef __SYNTHESIS__
//     if (line_tracing_spmspv) {
//         std::cout << "INFO: [Kernel SpMSpV] Checkout Results complete" << std::endl << std::flush;
//     }
//     #endif

//     write_back_gmem(
//         CR_to_WB_stream,
//         CR_to_WB_npld_stream,
//         mask,
//         result,
//         result_Nnz,
//         result_Nnz_incr,
//         zero,
//         mask_type
//     );

//     #ifndef __SYNTHESIS__
//     if (line_tracing_spmspv) {
//         std::cout << "INFO: [Kernel SpMSpV] Result writeback complete" << std::endl << std::flush;
//     }
//     #endif
// }


void kernel_spmspv(
    const SPMSPV_MAT_PKT_T *matrix,
    const IDX_T *mat_indptr,
    const IDX_T *mat_partptr,
    const IDX_VAL_T *vector,
    const VAL_T *mask,
    IDX_VAL_T *result,
    IDX_T num_rows,
    IDX_T num_cols,
    OP_T Op,
    MASK_T mask_type
) {
    #pragma HLS inline off

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

    // result Nnz counter
    IDX_T result_Nnz = 0;

    // total number of parts
    IDX_T num_parts = (num_rows + SPMSPV_OUT_BUF_LEN - 1) / SPMSPV_OUT_BUF_LEN;

    // number of rows in the last part
    IDX_T num_rows_last_part = (num_rows % SPMSPV_OUT_BUF_LEN) ? (num_rows % SPMSPV_OUT_BUF_LEN) : SPMSPV_OUT_BUF_LEN;

    // loop over parts
    loop_over_parts:
    for (unsigned int part_id = 0; part_id < num_parts; part_id++) {
        #pragma HLS pipeline off
        IDX_T num_rows_this_part = (part_id == (num_parts - 1)) ? num_rows_last_part : SPMSPV_OUT_BUF_LEN;
        IDX_T mat_indptr_base = (num_cols + 1) * part_id;
        IDX_T mat_row_id_base = SPMSPV_OUT_BUF_LEN * part_id;
        IDX_T result_Nnz_incr;
        #ifndef __SYNTHESIS__
        if (line_tracing_spmspv) {
            std::cout << "INFO: [Kernel SpMSpV] Partition " << part_id <<" start" << std::endl << std::flush;
            std::cout << "  # of rows this part: " << num_rows_this_part << std::endl << std::flush;
            std::cout << "          row id base: " << mat_row_id_base << std::endl << std::flush;
            std::cout << "          indptr base: " << mat_indptr_base << std::endl << std::flush;
        }
        #endif
        // loop_reset_output_buffer:
        // for (int i = 0; i < (num_rows_this_part + SPMV_NUM_PE_TOTAL - 1) / SPMV_NUM_PE_TOTAL; i++) {
        //     #pragma HLS UNROLL factor=2
        //     for (int c = 0; c < NUM_HBM_CHANNEL; c++) {
        //         #pragma HLS UNROLL
        //         for (int PE_idx = 0; PE_idx < PACK_SIZE; PE_idx++) {
        //             #pragma HLS UNROLL
        //             output_buffer[c][PE_idx][i] = Zero;
        //         }
        //     }
        // }
        spmspv_core(
            matrix,
            mat_indptr,
            mat_partptr,
            vector,
            mask,
            result,
            vec_num_nnz,
            mat_indptr_base,
            mat_row_id_base,
            part_id,
            result_Nnz,
            result_Nnz_incr,
            Op,
            Zero,
            mask_type,
            (num_rows_this_part + PACK_SIZE - 1) / PACK_SIZE
        );
        result_Nnz += result_Nnz_incr;
        #ifndef __SYNTHESIS__
        if (line_tracing_spmspv) {
            std::cout << "INFO: [Kernel SpMSpV] Partition " << part_id
                      << " complete" << std::endl << std::flush;
            std::cout << "     Nnz written back: " << result_Nnz << std::endl << std::flush;
        }
        #endif
    }

    // attach head
    IDX_VAL_T result_head;
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
