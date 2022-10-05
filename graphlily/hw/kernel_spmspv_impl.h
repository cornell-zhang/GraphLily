#include "libfpga/hisparse.h"
#include "libfpga/shuffle.h"
#include "libfpga/pe.h"

#include <ap_fixed.h>
#include <hls_stream.h>

#ifndef __SYNTHESIS__
#include <iostream>
#include <iomanip>
static bool line_tracing_spmspv = false;
static bool line_tracing_spmspv_write_back = false;
#endif

// vector loader for spmspv
static void load_vector_from_gmem(
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
static void load_matrix_from_gmem(
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
            #pragma HLS unroll
            // Use `unroll` here instead of `pipeline`, since the latter would
            // cause a weird issue in HLS synthesis stage with only log saying
            // `Encountered problem during source synthesis, Pre-synthesis
            // failed` and no more details provided. It emerges in Vitis 2022.1.
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

// write back to gemm
static void write_back_results (
    hls::stream<VEC_PLD_T> PE_to_WB_stream[PACK_SIZE],
    IDX_VAL_T *res_out,
    const VAL_T *mask,
    IDX_T mat_row_id_base,
    IDX_T &Nnz,
    MASK_T mask_type
) {
    IDX_T res_idx = Nnz;
    bool exit = false;
    char current_input = 0; // read from multiple PE output streams
    ap_uint<PACK_SIZE> finished = 0;

    spmspv_write_back_loop:
    while (!exit) {
        #pragma HLS pipeline II=1
        #pragma HLS dependence variable=res_out inter false

        VEC_PLD_T pld;
        if (!finished[current_input] && PE_to_WB_stream[current_input].read_nb(pld)) {
            if (pld.inst == EOS) {
                finished[current_input] = true;
            } else if (pld.inst != SOD && pld.inst != EOD) {
                IDX_T index = mat_row_id_base + pld.idx;
                bool do_write = false;
                switch (mask_type) {
                    case NOMASK:
                        do_write = true;
                        break;
                    case WRITETOONE:
                        do_write = (mask[index] != 0);
                        break;
                    case WRITETOZERO:
                        do_write = (mask[index] == 0);
                        break;
                    default:
                        do_write = false;
                        break;
                }
                if (do_write) {
                    IDX_VAL_T res_pld;
                    res_pld.index = index;
                    res_pld.val = pld.val;
                    res_idx++;
                    res_out[res_idx] = res_pld;
                    #ifndef __SYNTHESIS__
                    if (line_tracing_spmspv_write_back) {
                        std::cout << "INFO: [kernel SpMSpV] Write results"
                                  << " non-zero " << pld.val
                                  << " found at " << pld.idx
                                  << " mapped to " << index << std::endl << std::flush;
                    }
                    #endif
                }
            }
        }

        exit = finished.and_reduce();

        if ( (++current_input) == PACK_SIZE) {
            current_input = 0;
        }
    }
    Nnz = res_idx;
}

// load vec/mat -> shuffle -> pe compute -> writeback
static void spmspv_core(
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
    IDX_T &Nnz,
    OP_T Op,
    VAL_T Zero,
    MASK_T mask_type,
    const unsigned used_buf_len_per_pe
) {
    // fifos
    hls::stream<IDX_VAL_T> VL_to_ML_stream;
    #pragma HLS stream variable=VL_to_ML_stream depth=FIFO_DEPTH

    hls::stream<UPDATE_PLD_T> DL_to_SF_stream[PACK_SIZE];
    hls::stream<UPDATE_PLD_T> SF_to_PE_stream[PACK_SIZE];
    hls::stream<VEC_PLD_T> PE_to_WB_stream[PACK_SIZE];
    #pragma HLS stream variable=DL_to_SF_stream depth=FIFO_DEPTH
    #pragma HLS stream variable=SF_to_PE_stream depth=FIFO_DEPTH
    #pragma HLS stream variable=PE_to_WB_stream depth=FIFO_DEPTH
    #pragma HLS bind_storage variable=DL_to_SF_stream type=FIFO impl=SRL
    #pragma HLS bind_storage variable=SF_to_PE_stream type=FIFO impl=SRL
    #pragma HLS bind_storage variable=PE_to_WB_stream type=FIFO impl=SRL

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

    pe_bram_sparse<0, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[0],
        PE_to_WB_stream[0],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram_sparse<1, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[1],
        PE_to_WB_stream[1],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram_sparse<2, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[2],
        PE_to_WB_stream[2],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram_sparse<3, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[3],
        PE_to_WB_stream[3],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram_sparse<4, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[4],
        PE_to_WB_stream[4],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram_sparse<5, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[5],
        PE_to_WB_stream[5],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram_sparse<6, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
        SF_to_PE_stream[6],
        PE_to_WB_stream[6],
        used_buf_len_per_pe,
        Op,
        Zero
    );
    pe_bram_sparse<7, SPMSPV_OUT_BUF_LEN / PACK_SIZE, PACK_SIZE>(
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
        mask_type
    );

    #ifndef __SYNTHESIS__
    if (line_tracing_spmspv) {
        std::cout << "INFO: [Kernel SpMSpV] Result writeback complete" << std::endl << std::flush;
    }
    #endif
}

static void kernel_spmspv(
    const SPMSPV_MAT_PKT_T *matrix,
    const IDX_T *mat_indptr,
    const IDX_T *mat_partptr,
    const IDX_VAL_T *vector,
    const VAL_T *mask,
    IDX_VAL_T *result,
    IDX_T num_rows,
    IDX_T num_cols,
    OP_T Op,
    MASK_T mask_type,
    VAL_T Zero
) {
    #pragma HLS inline off

    IDX_T vec_num_nnz = vector[0].index;

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
        #ifndef __SYNTHESIS__
        if (line_tracing_spmspv) {
            std::cout << "INFO: [Kernel SpMSpV] Partition " << part_id <<" start" << std::endl << std::flush;
            std::cout << "  # of rows this part: " << num_rows_this_part << std::endl << std::flush;
            std::cout << "          row id base: " << mat_row_id_base << std::endl << std::flush;
            std::cout << "          indptr base: " << mat_indptr_base << std::endl << std::flush;
        }
        #endif
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
            Op,
            Zero,
            mask_type,
            (num_rows_this_part + PACK_SIZE - 1) / PACK_SIZE
        );
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
