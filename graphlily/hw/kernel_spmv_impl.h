#include "./overlay.h"

#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <iomanip>

#include <hls_stream.h>
#include <ap_fixed.h>

#include "./util.h"
#include "./shuffle.h"
#include "./ufixed_pe_fwd.h"
// #include "./float_pe.h"

#ifndef __SYNTHESIS__
static bool line_tracing_spmv = false;
#endif

typedef struct shuffle_1_inout_value_type {
    IDX_T row_idx;
    VAL_T mat_val;
} SF_1_IO_VAL_T;

typedef struct shuffle_1_inout_type {
    IDX_T index;
    SF_1_IO_VAL_T data;
} SF_1_IO_T;

typedef struct shuffle_2_inout_value_type {
    VAL_T vec_val;
    VAL_T mat_val;
} SF_2_IO_VAL_T;

typedef struct shuffle_2_inout_type {
    IDX_T index;
    SF_2_IO_VAL_T data;
} SF_2_IO_T;


// ML (matrix loader). Load matrix from one HBM channel and unpack to streams.
void matrix_loader_hbm_to_stream_one_channel(
    const SPMV_MAT_PKT_T *matrix_one_channel,             // in
    unsigned partition_idx,                               // in
    unsigned num_partitions,                              // in
    hls::stream<SF_1_IO_T> ML_to_SF_1_stream[PACK_SIZE],  // out
    hls::stream<unsigned> &num_payloads                   // out
) {
    IDX_T partition_start = matrix_one_channel[2*partition_idx].indices.data[0];
    PACKED_IDX_T partition_size = matrix_one_channel[2*partition_idx + 1].indices;

    unsigned max_size = array_max<unsigned, PACK_SIZE>(partition_size.data);

    unsigned payload_count[PACK_SIZE];
    #pragma HLS ARRAY_PARTITION variable=payload_count complete
    for (int k = 0; k < PACK_SIZE; k++) {
        #pragma HLS UNROLL
        payload_count[k] = 0;
    }

    IDX_T row_idx[PACK_SIZE];
    #pragma HLS ARRAY_PARTITION variable=row_idx complete
    for (int k = 0; k < PACK_SIZE; k++) {
        #pragma HLS UNROLL
        row_idx[k] = k;
    }

    // Burst read
    loop_matrix_loader_hbm_to_stream_one_channel:
    for (int i = 0; i < max_size; i++) {
        #pragma HLS PIPELINE II=1
        SPMV_MAT_PKT_T mat_pkt = matrix_one_channel[i + partition_start + 2*num_partitions];
        for (unsigned k = 0; k < PACK_SIZE; k++) {
            #pragma HLS UNROLL
            if (i < partition_size.data[k]) {
                if (mat_pkt.indices.data[k] == IDX_MARKER) {
                     // Be careful: mat_pkt.vals.data[k] can not be larger than power(2, 16)
                    row_idx[k] += (PACK_SIZE * (unsigned)mat_pkt.vals.data[k]);
                    // row_idx[k] += PACK_SIZE;  // Row index within each packed stream of rows
                } else {
                    SF_1_IO_T input_to_SF_1;
                    input_to_SF_1.index = mat_pkt.indices.data[k];
                    input_to_SF_1.data = (SF_1_IO_VAL_T){row_idx[k], mat_pkt.vals.data[k]};
                    ML_to_SF_1_stream[k].write(input_to_SF_1);
                    payload_count[k]++;

                    #ifndef __SYNTHESIS__
                    if (line_tracing_spmv) {
                        std::cout << "PE_idx: " << k << std::endl;
                        std::cout << "input_to_SF_1: " << input_to_SF_1.index << " "
                                                       << input_to_SF_1.data.row_idx << " "
                                                       << float(input_to_SF_1.data.mat_val) << " "
                                                       << std::endl;
                    }
                    #endif
                }
            }
        }
    }

    num_payloads.write(array_sum<unsigned, PACK_SIZE>(payload_count));
}


// Load vector from DDR to URAM.
void vector_loader_ddr_to_uram(
    const PACKED_VAL_T *vector,                                           // in
    unsigned num_cols,                                                    // in
    unsigned col_partition_idx,                                           // in
    unsigned num_col_partitions,                                          // in
    VAL_T vector_uram[NUM_HBM_CHANNEL]
        [NUM_BANK_PER_HBM_CHANNEL][VEC_BUF_LEN/NUM_BANK_PER_HBM_CHANNEL]  // out
) {
    unsigned size = VEC_BUF_LEN;
    if (col_partition_idx == (num_col_partitions - 1)) {
        size = num_cols - (num_col_partitions - 1) * VEC_BUF_LEN;
    }
    assert(size % PACK_SIZE == 0);
    unsigned vsize = size / PACK_SIZE;

    PACKED_VAL_T tmp_duplicate[NUM_HBM_CHANNEL];
    #pragma HLS array_partition variable=tmp_duplicate complete

    int i_duplicate[NUM_HBM_CHANNEL];
    #pragma HLS array_partition variable=i_duplicate complete

    // Burst read
    loop_vector_loader_ddr_to_uram:
    for (int i = 0; i < vsize; i++) {
        #pragma HLS PIPELINE II=1
        PACKED_VAL_T tmp = vector[col_partition_idx * VEC_BUF_LEN / PACK_SIZE + i];
        for (int j = 0; j < NUM_HBM_CHANNEL; j++) {
            #pragma HLS UNROLL
            tmp_duplicate[j] = HLS_REG(tmp);
            i_duplicate[j] = HLS_REG(i);
            for (int k = 0; k < PACK_SIZE; k++) {
                #pragma HLS UNROLL
                vector_uram[j][k % NUM_BANK_PER_HBM_CHANNEL][i_duplicate[j] * NUM_PORT_PER_BANK
                    + k / NUM_BANK_PER_HBM_CHANNEL] = tmp_duplicate[j].data[k];
            }
        }
    }
}


// VL (vector loader). Load vector from URAM to stream.
void vector_loader_uram_to_stream_one_channel(
    VAL_T vector_uram_one_channel[NUM_BANK_PER_HBM_CHANNEL][VEC_BUF_LEN/NUM_BANK_PER_HBM_CHANNEL],  // in
    hls::stream<SF_1_IO_T> SF_1_to_VL_stream[PACK_SIZE],                                            // in
    hls::stream<SF_2_IO_T> VL_to_SF_2_stream[PACK_SIZE],                                            // out
    hls::stream<unsigned> &num_payloads_in,                                                         // in
    hls::stream<unsigned> &num_payloads_out                                                         // out
) {
    // TODO: now we assume PACK_SIZE == NUM_BANK_PER_HBM_CHANNEL; need to fix later

    unsigned num_payloads;
    bool prev_finish = false;
    bool fifo_allempty = false;

    bool fifo_empty[PACK_SIZE];
    #pragma HLS array_partition variable=fifo_empty complete

    loop_vector_loader_uram_to_stream_one_channel:
    while (!(prev_finish && fifo_allempty)) {
        #pragma HLS pipeline II=1
        if (!prev_finish) { prev_finish = num_payloads_in.read_nb(num_payloads); }

        for (unsigned PE_idx = 0; PE_idx < PACK_SIZE; PE_idx++) {
            #pragma HLS unroll
            SF_1_IO_T payload_in;
            if (SF_1_to_VL_stream[PE_idx].read_nb(payload_in)) {
                fifo_empty[PE_idx] = false;
                SF_2_IO_T payload_out;
                payload_out.index = payload_in.data.row_idx;
                payload_out.data = (SF_2_IO_VAL_T){vector_uram_one_channel[PE_idx][payload_in.index
                    / PACK_SIZE], payload_in.data.mat_val};
                VL_to_SF_2_stream[PE_idx].write(payload_out);

                #ifndef __SYNTHESIS__
                if (line_tracing_spmv) {
                    std::cout << "PE_idx: " << PE_idx << std::endl;
                    std::cout << "payload_in.index:: " << payload_in.index << std::endl;
                    std::cout << "payload_out: " << payload_out.index << " "
                                                 << float(payload_out.data.vec_val) << " "
                                                 << float(payload_out.data.mat_val) << " "
                                                 << std::endl;
                }
                #endif

            } else {
                fifo_empty[PE_idx] = true;
            }
        }
        fifo_allempty = array_and_reduction<PACK_SIZE>(fifo_empty);
    }

    num_payloads_out.write(num_payloads);
}


void compute_spmv_one_channel(
    const SPMV_MAT_PKT_T *matrix_one_channel,                                                       // in
    unsigned partition_idx,                                                                         // in
    unsigned num_partitions,                                                                        // in
    VAL_T Zero,                                                                                     // in
    VAL_T vector_uram_one_channel[NUM_BANK_PER_HBM_CHANNEL][VEC_BUF_LEN/NUM_BANK_PER_HBM_CHANNEL],  // in
    OP_T Op,                                                                                        // in
    VAL_T out_uram[PACK_SIZE][SPMV_OUT_BUF_LEN / SPMV_NUM_PE_TOTAL]                                      // out
) {
    // FIFOs
    hls::stream<unsigned> ML_to_SF_1_num_payloads_stream;
    hls::stream<unsigned> SF_1_to_VL_num_payloads_stream;
    hls::stream<unsigned> VL_to_SF_2_num_payloads_stream;
    hls::stream<unsigned> SF_2_to_PE_num_payloads_stream;
    #pragma HLS stream variable=ML_to_SF_1_num_payloads_stream depth=2
    #pragma HLS stream variable=SF_1_to_VL_num_payloads_stream depth=2
    #pragma HLS stream variable=VL_to_SF_2_num_payloads_stream depth=2
    #pragma HLS stream variable=SF_2_to_PE_num_payloads_stream depth=2

    hls::stream<SF_1_IO_T> ML_to_SF_1_stream[PACK_SIZE];
    hls::stream<SF_1_IO_T> SF_1_to_VL_stream[PACK_SIZE];
    hls::stream<SF_2_IO_T> VL_to_SF_2_stream[PACK_SIZE];
    hls::stream<SF_2_IO_T> SF_2_to_PE_stream[PACK_SIZE];
    #pragma HLS stream variable=ML_to_SF_1_stream depth=FIFO_DEPTH
    #pragma HLS stream variable=SF_1_to_VL_stream depth=FIFO_DEPTH
    #pragma HLS stream variable=VL_to_SF_2_stream depth=FIFO_DEPTH
    #pragma HLS stream variable=SF_2_to_PE_stream depth=FIFO_DEPTH
    #pragma HLS RESOURCE variable=ML_to_SF_1_stream core=FIFO_SRL
    #pragma HLS RESOURCE variable=SF_1_to_VL_stream core=FIFO_SRL
    #pragma HLS RESOURCE variable=VL_to_SF_2_stream core=FIFO_SRL
    #pragma HLS RESOURCE variable=SF_2_to_PE_stream core=FIFO_SRL

    // Dataflow pipeline
    {
        #pragma HLS dataflow
        matrix_loader_hbm_to_stream_one_channel(
            matrix_one_channel,
            partition_idx,
            num_partitions,
            ML_to_SF_1_stream,
            ML_to_SF_1_num_payloads_stream
        );

        #ifndef __SYNTHESIS__
        if (line_tracing_spmv) {
            std::cout << "INFO : [Kernel SpMV] Matrix Loader complete" << std::endl << std::flush;
        }
        #endif

        shuffler_1p<SF_1_IO_T, SF_1_IO_VAL_T, PACK_SIZE, PACK_SIZE, BANK_ID_MASK>(
            ML_to_SF_1_stream,
            SF_1_to_VL_stream,
            ML_to_SF_1_num_payloads_stream,
            SF_1_to_VL_num_payloads_stream
        );

        #ifndef __SYNTHESIS__
        if (line_tracing_spmv) {
            std::cout << "INFO : [Kernel SpMV] Shuffler 1 complete" << std::endl << std::flush;
        }
        #endif

        vector_loader_uram_to_stream_one_channel(
            vector_uram_one_channel,
            SF_1_to_VL_stream,
            VL_to_SF_2_stream,
            SF_1_to_VL_num_payloads_stream,
            VL_to_SF_2_num_payloads_stream
        );

        #ifndef __SYNTHESIS__
        if (line_tracing_spmv) {
            std::cout << "INFO : [Kernel SpMV] Vector Loader complete" << std::endl << std::flush;
        }
        #endif

        shuffler_1p<SF_2_IO_T, SF_2_IO_VAL_T, PACK_SIZE, PACK_SIZE, BANK_ID_MASK>(
            VL_to_SF_2_stream,
            SF_2_to_PE_stream,
            VL_to_SF_2_num_payloads_stream,
            SF_2_to_PE_num_payloads_stream
        );

        #ifndef __SYNTHESIS__
        if (line_tracing_spmv) {
            std::cout << "INFO : [Kernel SpMV] Shuffler 2 complete" << std::endl << std::flush;
        }
        #endif

        ufixed_pe_cluster_uram<VAL_T, OP_T, SF_2_IO_T, PACK_SIZE, BANK_ID_NBITS, SPMV_OUT_BUF_LEN / SPMV_NUM_PE_TOTAL>(
            SF_2_to_PE_stream,
            out_uram,
            Op,
            Zero,
            SF_2_to_PE_num_payloads_stream
        );

        // float_pe_cluster_uram<VAL_T, OP_T, SF_2_IO_T, PACK_SIZE, BANK_ID_NBITS, SPMV_OUT_BUF_LEN / SPMV_NUM_PE_TOTAL>(
        //     SF_2_to_PE_stream,
        //     out_uram,
        //     Op,
        //     Zero,
        //     SF_2_to_PE_num_payloads_stream
        // );

        #ifndef __SYNTHESIS__
        if (line_tracing_spmv) {
            std::cout << "INFO : [Kernel SpMV] Process Elements complete" << std::endl << std::flush;
        }
        #endif
    }

    #ifndef __SYNTHESIS__
    if (line_tracing_spmv) {
        std::cout << "INFO : [Kernel SpMV] Computation complete" << std::endl << std::flush;
    }
    #endif
}


void write_to_out_ddr(
    const VAL_T out_uram[NUM_HBM_CHANNEL][PACK_SIZE][SPMV_OUT_BUF_LEN / SPMV_NUM_PE_TOTAL],  // in
    unsigned row_partition_idx,                                                              // in
    unsigned num_row_partitions,                                                             // in
    unsigned num_rows,                                                                       // in
    const PACKED_VAL_T *mask,                                                                // in
    MASK_T mask_type,                                                                        // in
    PACKED_VAL_T *out                                                                        // out
) {
    unsigned size = SPMV_OUT_BUF_LEN;
    if (row_partition_idx == (num_row_partitions - 1)) {
        size = num_rows - (num_row_partitions - 1) * SPMV_OUT_BUF_LEN;
    }
    assert(size % PACK_SIZE == 0);
    unsigned vsize = size / PACK_SIZE;

    PACKED_VAL_T tmp_out;
    PACKED_VAL_T tmp_mask;

    loop_write_to_out_ddr:
    for (int i = 0; i < vsize; i++) {
        #pragma HLS PIPELINE II=1
        if (mask_type != NOMASK) {
            tmp_mask = mask[i + row_partition_idx * SPMV_OUT_BUF_LEN / PACK_SIZE];
        }
        for (int k = 0; k < PACK_SIZE; k++) {
            #pragma HLS UNROLL
            VAL_T val = out_uram[i%NUM_HBM_CHANNEL][k][i/NUM_HBM_CHANNEL];
            if (mask_type == NOMASK) {
                tmp_out.data[k] = val;
            } else if (mask_type == WRITETOZERO) {
                if (tmp_mask.data[k] == 0) {
                    tmp_out.data[k] = val;
                } else {
                    tmp_out.data[k] = 0;
                }
            } else if (mask_type == WRITETOONE) {
                if (tmp_mask.data[k] == 0) {
                    tmp_out.data[k] = 0;
                } else {
                    tmp_out.data[k] = val;
                }
            } else {
                std::cout << "Invalid mask type" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        out[i + row_partition_idx * SPMV_OUT_BUF_LEN / PACK_SIZE] = tmp_out;
    }
}


void kernel_spmv(
#if (NUM_HBM_CHANNEL >= 1)
    const SPMV_MAT_PKT_T *channel_0_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 2)
    const SPMV_MAT_PKT_T *channel_1_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 4)
    const SPMV_MAT_PKT_T *channel_2_matrix,
    const SPMV_MAT_PKT_T *channel_3_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 8)
    const SPMV_MAT_PKT_T *channel_4_matrix,
    const SPMV_MAT_PKT_T *channel_5_matrix,
    const SPMV_MAT_PKT_T *channel_6_matrix,
    const SPMV_MAT_PKT_T *channel_7_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 16)
    const SPMV_MAT_PKT_T *channel_8_matrix,
    const SPMV_MAT_PKT_T *channel_9_matrix,
    const SPMV_MAT_PKT_T *channel_10_matrix,
    const SPMV_MAT_PKT_T *channel_11_matrix,
    const SPMV_MAT_PKT_T *channel_12_matrix,
    const SPMV_MAT_PKT_T *channel_13_matrix,
    const SPMV_MAT_PKT_T *channel_14_matrix,
    const SPMV_MAT_PKT_T *channel_15_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 32)
    const SPMV_MAT_PKT_T *channel_16_matrix,
    const SPMV_MAT_PKT_T *channel_17_matrix,
    const SPMV_MAT_PKT_T *channel_18_matrix,
    const SPMV_MAT_PKT_T *channel_19_matrix,
    const SPMV_MAT_PKT_T *channel_20_matrix,
    const SPMV_MAT_PKT_T *channel_21_matrix,
    const SPMV_MAT_PKT_T *channel_22_matrix,
    const SPMV_MAT_PKT_T *channel_23_matrix,
    const SPMV_MAT_PKT_T *channel_24_matrix,
    const SPMV_MAT_PKT_T *channel_25_matrix,
    const SPMV_MAT_PKT_T *channel_26_matrix,
    const SPMV_MAT_PKT_T *channel_27_matrix,
    const SPMV_MAT_PKT_T *channel_28_matrix,
    const SPMV_MAT_PKT_T *channel_29_matrix,
    const SPMV_MAT_PKT_T *channel_30_matrix,
    const SPMV_MAT_PKT_T *channel_31_matrix,
#endif
    const PACKED_VAL_T *vector,
    const PACKED_VAL_T *mask,
    PACKED_VAL_T *out,
    unsigned num_rows,
    unsigned num_cols,
    OP_T Op,
    MASK_T mask_type,
    VAL_T out_uram[NUM_HBM_CHANNEL][PACK_SIZE][SPMV_OUT_BUF_LEN / SPMV_NUM_PE_TOTAL]
) {
    #pragma HLS inline off

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

    // All PEs within the same channel share one vector buffer.
    VAL_T vector_uram[NUM_HBM_CHANNEL][NUM_BANK_PER_HBM_CHANNEL][VEC_BUF_LEN / NUM_BANK_PER_HBM_CHANNEL];
    #pragma HLS RESOURCE variable=vector_uram core=XPM_MEMORY uram
    #pragma HLS ARRAY_PARTITION variable=vector_uram complete dim=1
    #pragma HLS ARRAY_PARTITION variable=vector_uram complete dim=2

    unsigned num_row_partitions = (num_rows + SPMV_OUT_BUF_LEN - 1) / SPMV_OUT_BUF_LEN;
    unsigned num_col_partitions = (num_cols + VEC_BUF_LEN - 1) / VEC_BUF_LEN;
    unsigned num_partitions = num_row_partitions * num_col_partitions;

    // Iterate row partitions
    for (int row_partition_idx = 0; row_partition_idx < num_row_partitions; row_partition_idx++) {

        // TODO: is there a faster way to reset bram/uram?
        unsigned out_buf_len = SPMV_OUT_BUF_LEN;
        if (row_partition_idx == (num_row_partitions - 1)) {
            out_buf_len = num_rows - (num_row_partitions - 1) * SPMV_OUT_BUF_LEN;
        }
        loop_initialize_out_uram:
        for (int i = 0; i < out_buf_len / SPMV_NUM_PE_TOTAL; i++) {
            #pragma HLS UNROLL factor=2
            for (int c = 0; c < NUM_HBM_CHANNEL; c++) {
                #pragma HLS UNROLL
                for (int PE_idx = 0; PE_idx < PACK_SIZE; PE_idx++) {
                    #pragma HLS UNROLL
                    out_uram[c][PE_idx][i] = Zero;
                }
            }
        }

        // Iterate column partitions
        for (int col_partition_idx = 0; col_partition_idx < num_col_partitions; col_partition_idx++) {
            #pragma HLS dataflow

            unsigned partition_idx = row_partition_idx * num_col_partitions + col_partition_idx;

            #ifndef __SYNTHESIS__
            if (line_tracing_spmv) {
                std::cout << "row_partition_idx: " << row_partition_idx << std::endl;
                std::cout << "col_partition_idx: " << col_partition_idx << std::endl;
            }
            #endif

            vector_loader_ddr_to_uram(
                vector,
                num_cols,
                col_partition_idx,
                num_col_partitions,
                vector_uram
            );

#if (NUM_HBM_CHANNEL >= 1)
            compute_spmv_one_channel(
                channel_0_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[0],
                Op,
                out_uram[0]
            );
#endif
#if (NUM_HBM_CHANNEL >= 2)
            compute_spmv_one_channel(
                channel_1_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[1],
                Op,
                out_uram[1]
            );
#endif
#if (NUM_HBM_CHANNEL >= 4)
            compute_spmv_one_channel(
                channel_2_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[2],
                Op,
                out_uram[2]
            );
            compute_spmv_one_channel(
                channel_3_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[3],
                Op,
                out_uram[3]
            );
#endif
#if (NUM_HBM_CHANNEL >= 8)
            compute_spmv_one_channel(
                channel_4_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[4],
                Op,
                out_uram[4]
            );
            compute_spmv_one_channel(
                channel_5_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[5],
                Op,
                out_uram[5]
            );
            compute_spmv_one_channel(
                channel_6_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[6],
                Op,
                out_uram[6]
            );
            compute_spmv_one_channel(
                channel_7_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[7],
                Op,
                out_uram[7]
            );
#endif
#if (NUM_HBM_CHANNEL >= 16)
            compute_spmv_one_channel(
                channel_8_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[8],
                Op,
                out_uram[8]
            );
            compute_spmv_one_channel(
                channel_9_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[9],
                Op,
                out_uram[9]
            );
            compute_spmv_one_channel(
                channel_10_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[10],
                Op,
                out_uram[10]
            );
            compute_spmv_one_channel(
                channel_11_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[11],
                Op,
                out_uram[11]
            );
            compute_spmv_one_channel(
                channel_12_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[12],
                Op,
                out_uram[12]
            );
            compute_spmv_one_channel(
                channel_13_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[13],
                Op,
                out_uram[13]
            );
            compute_spmv_one_channel(
                channel_14_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[14],
                Op,
                out_uram[14]
            );
            compute_spmv_one_channel(
                channel_15_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[15],
                Op,
                out_uram[15]
            );
#endif
#if (NUM_HBM_CHANNEL >= 32)
            compute_spmv_one_channel(
                channel_16_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[16],
                Op,
                out_uram[16]
            );
            compute_spmv_one_channel(
                channel_17_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[17],
                Op,
                out_uram[17]
            );
            compute_spmv_one_channel(
                channel_18_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[18],
                Op,
                out_uram[18]
            );
            compute_spmv_one_channel(
                channel_19_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[19],
                Op,
                out_uram[19]
            );
            compute_spmv_one_channel(
                channel_20_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[20],
                Op,
                out_uram[20]
            );
            compute_spmv_one_channel(
                channel_21_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[21],
                Op,
                out_uram[21]
            );
            compute_spmv_one_channel(
                channel_22_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[22],
                Op,
                out_uram[22]
            );
            compute_spmv_one_channel(
                channel_23_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[23],
                Op,
                out_uram[23]
            );
            compute_spmv_one_channel(
                channel_24_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[24],
                Op,
                out_uram[24]
            );
            compute_spmv_one_channel(
                channel_25_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[25],
                Op,
                out_uram[25]
            );
            compute_spmv_one_channel(
                channel_26_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[26],
                Op,
                out_uram[26]
            );
            compute_spmv_one_channel(
                channel_27_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[27],
                Op,
                out_uram[27]
            );
            compute_spmv_one_channel(
                channel_28_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[28],
                Op,
                out_uram[28]
            );
            compute_spmv_one_channel(
                channel_29_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[29],
                Op,
                out_uram[29]
            );
            compute_spmv_one_channel(
                channel_30_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[30],
                Op,
                out_uram[30]
            );
            compute_spmv_one_channel(
                channel_31_matrix,
                partition_idx,
                num_partitions,
                Zero,
                vector_uram[31],
                Op,
                out_uram[31]
            );
#endif
        }

        write_to_out_ddr(out_uram, row_partition_idx, num_row_partitions, num_rows, mask, mask_type, out);
    }
}
