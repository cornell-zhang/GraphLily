#include "kernel_spmv_v3.h"
#include <hls_stream.h>


static void read_data_one_channel(const packed_index_t *indices,
                                  const packed_data_t *vals,
                                  hls::stream<packed_index_t> &indices_stream,
                                  hls::stream<packed_data_t> &vals_stream,
                                  unsigned int start,
                                  unsigned int end) {
    // Pass size to compute_spmv_one_channel by indices_stream
    unsigned int size = end - start;
    packed_index_t tmp;
    tmp.data[0] = size;
    indices_stream << tmp;
    // Burst read
    loop_read_data_one_channel:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        indices_stream << indices[i + start];
        vals_stream << vals[i + start];
    }
}


static void read_vector(const data_t *vector,
                        data_t channel_0_vector_one_partition_bram[NUM_PE_PER_HBM_CHANNEL][VECTOR_BUFFER_LEN],
                        data_t channel_1_vector_one_partition_bram[NUM_PE_PER_HBM_CHANNEL][VECTOR_BUFFER_LEN],
                        unsigned int num_cols,
                        unsigned int num_col_partitions,
                        unsigned int partition_idx) {
    unsigned int size = VECTOR_BUFFER_LEN;
    if (partition_idx == (num_col_partitions - 1))
        size = num_cols - (num_col_partitions - 1) * VECTOR_BUFFER_LEN;
    // Burst read
    loop_read_vector:
    data_t tmp;
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        tmp = vector[partition_idx * VECTOR_BUFFER_LEN + i];
        for (int j = 0; j < NUM_PE_PER_HBM_CHANNEL; j++) {
            #pragma HLS UNROLL
            channel_0_vector_one_partition_bram[j][i] = tmp;
            channel_1_vector_one_partition_bram[j][i] = tmp;
        }
    }
}


static void compute_spmv_one_channel(hls::stream<packed_index_t> &indices_stream,
                                     hls::stream<packed_data_t> &vals_stream,
                                     data_t vector_one_partition_bram[NUM_PE_PER_HBM_CHANNEL][VECTOR_BUFFER_LEN],
                                     hls::stream<data_t> out_stream[NUM_PE_PER_HBM_CHANNEL],
                                     unsigned int channel_idx) {
    unsigned int index[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=index complete

    data_t val[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=val complete

    data_t tmp_out[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=tmp_out complete

    unsigned int row_idx[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=row_idx complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        tmp_out[PE_idx] = 0;
    }

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        row_idx[PE_idx] = PE_idx + channel_idx * NUM_PE_PER_HBM_CHANNEL;
    }

    unsigned int size = indices_stream.read().data[0];

    loop_compute_spmv_one_channel:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1

        packed_index_t packed_indices = indices_stream.read();
        packed_data_t packed_vals = vals_stream.read();

        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL

            index[PE_idx] = packed_indices.data[PE_idx];
            val[PE_idx] = packed_vals.data[PE_idx];

            if (index[PE_idx] == IDX_MARKER) {
                out_stream[PE_idx] << tmp_out[PE_idx];
                tmp_out[PE_idx] = 0;
                row_idx[PE_idx] += NUM_PE_TOTAL;
            } else {
                tmp_out[PE_idx] += val[PE_idx] * vector_one_partition_bram[PE_idx][index[PE_idx]];
            }
        }
    }
}


static void write_out_bram(hls::stream<data_t> channel_0_out_stream[NUM_PE_PER_HBM_CHANNEL],
                           hls::stream<data_t> channel_1_out_stream[NUM_PE_PER_HBM_CHANNEL],
                           data_t out_bram[MAX_NUM_ROWS],
                           const unsigned int num_rows) {
    loop_write_out_bram:
    for (int row_idx = 0; row_idx < num_rows; row_idx+=NUM_PE_TOTAL) {
        #pragma HLS PIPELINE II=1

        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL

            if ((row_idx + NUM_PE_PER_HBM_CHANNEL + PE_idx) < num_rows)
                out_bram[row_idx + NUM_PE_PER_HBM_CHANNEL + PE_idx] += channel_1_out_stream[PE_idx].read();
            if ((row_idx + PE_idx) < num_rows)
                out_bram[row_idx + PE_idx] += channel_0_out_stream[PE_idx].read();
        }
    }
}


extern "C" {

void kernel_spmv_v3(
    const data_t *vector,                            // The dense vector
    const unsigned int *channel_0_partition_indptr,  // Indptr of the partitions (the first channel)
    const packed_index_t *channel_0_indices,         // Indices (the first channel)
    const packed_data_t *channel_0_vals,             // Vals (the first channel)
    const unsigned int *channel_1_partition_indptr,  // Indptr of the partitions (the second channel)
    const packed_index_t *channel_1_indices,         // Indices (the second channel)
    const packed_data_t *channel_1_vals,             // Vals (the second channel)
    data_t *out,                                     // Output of the SpMV kernel
    unsigned int num_rows,                           // Number of rows of the sparse matrix
    unsigned int num_cols,                           // Number of columns of the sparse matrix
    unsigned int num_times                           // Running the kernel num_times for performance measurement
) {
#pragma HLS INTERFACE m_axi port=channel_0_indices          offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=channel_0_vals             offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=channel_1_indices          offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=channel_1_vals             offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=vector                     offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=channel_0_partition_indptr offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=channel_1_partition_indptr offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=out                        offset=slave bundle=gmem7

#pragma HLS INTERFACE s_axilite port=channel_0_indices          bundle=control
#pragma HLS INTERFACE s_axilite port=channel_0_vals             bundle=control
#pragma HLS INTERFACE s_axilite port=channel_1_indices          bundle=control
#pragma HLS INTERFACE s_axilite port=channel_1_vals             bundle=control
#pragma HLS INTERFACE s_axilite port=vector                     bundle=control
#pragma HLS INTERFACE s_axilite port=channel_0_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_1_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=out                        bundle=control

#pragma HLS INTERFACE s_axilite port=num_rows  bundle=control
#pragma HLS INTERFACE s_axilite port=num_cols  bundle=control
#pragma HLS INTERFACE s_axilite port=num_times bundle=control
#pragma HLS INTERFACE s_axilite port=return    bundle=control

#pragma HLS DATA_PACK variable=channel_0_indices
#pragma HLS DATA_PACK variable=channel_0_vals
#pragma HLS DATA_PACK variable=channel_1_indices
#pragma HLS DATA_PACK variable=channel_1_vals

    data_t channel_0_vector_one_partition_bram[NUM_PE_PER_HBM_CHANNEL][VECTOR_BUFFER_LEN];
    #pragma HLS ARRAY_PARTITION variable=channel_0_vector_one_partition_bram complete dim=1

    data_t channel_1_vector_one_partition_bram[NUM_PE_PER_HBM_CHANNEL][VECTOR_BUFFER_LEN];
    #pragma HLS ARRAY_PARTITION variable=channel_1_vector_one_partition_bram complete dim=1

    data_t out_bram[MAX_NUM_ROWS];
    #pragma HLS ARRAY_PARTITION variable=out_bram cyclic factor=NUM_PE_TOTAL
    // #pragma HLS RESOURCE variable=out_bram core=XPM_MEMORY uram
    // TODO: bank conflicts when writing out_stream to out_bram?

    hls::stream<packed_index_t> channel_0_indices_stream;
    hls::stream<packed_data_t> channel_0_vals_stream;
    #pragma HLS STREAM variable=channel_0_indices_stream depth=512
    #pragma HLS STREAM variable=channel_0_vals_stream depth=512

    hls::stream<packed_index_t> channel_1_indices_stream;
    hls::stream<packed_data_t> channel_1_vals_stream;
    #pragma HLS STREAM variable=channel_1_indices_stream depth=512
    #pragma HLS STREAM variable=channel_1_vals_stream depth=512

    hls::stream<data_t> channel_0_out_stream[NUM_PE_PER_HBM_CHANNEL];
    hls::stream<data_t> channel_1_out_stream[NUM_PE_PER_HBM_CHANNEL];
    /* Depth is set to the same for all the streams in one array */
    #pragma HLS STREAM variable=channel_0_out_stream depth=512
    #pragma HLS STREAM variable=channel_1_out_stream depth=512

    unsigned int num_col_partitions = (num_cols + VECTOR_BUFFER_LEN - 1) / VECTOR_BUFFER_LEN;

    // Running the same kernel num_times for performance measurement
    for (int count = 0; count < num_times; count++) {

        // Initialize out_bram to 0
        loop_initialize_out_bram:
        for (int PE_idx = 0; PE_idx < NUM_PE_TOTAL; PE_idx++) {
            #pragma HLS UNROLL
            for (int row_idx = 0; row_idx < num_rows; row_idx+=NUM_PE_TOTAL) {
                out_bram[row_idx + PE_idx] = 0;
            }
        }

        // Iterate column partitions
        for (int partition_idx = 0; partition_idx < num_col_partitions; partition_idx++) {
            #pragma HLS dataflow
            read_vector(vector,
                        channel_0_vector_one_partition_bram,
                        channel_1_vector_one_partition_bram,
                        num_cols,
                        num_col_partitions,
                        partition_idx);
            read_data_one_channel(channel_0_indices,
                                  channel_0_vals,
                                  channel_0_indices_stream,
                                  channel_0_vals_stream,
                                  channel_0_partition_indptr[partition_idx],
                                  channel_0_partition_indptr[partition_idx + 1]);
            read_data_one_channel(channel_1_indices,
                                  channel_1_vals,
                                  channel_1_indices_stream,
                                  channel_1_vals_stream,
                                  channel_1_partition_indptr[partition_idx],
                                  channel_1_partition_indptr[partition_idx + 1]);
            compute_spmv_one_channel(channel_0_indices_stream,
                                     channel_0_vals_stream,
                                     channel_0_vector_one_partition_bram,
                                     channel_0_out_stream,
                                     0);
            compute_spmv_one_channel(channel_1_indices_stream,
                                     channel_1_vals_stream,
                                     channel_1_vector_one_partition_bram,
                                     channel_1_out_stream,
                                     1);
            write_out_bram(channel_0_out_stream,
                           channel_1_out_stream,
                           out_bram,
                           num_rows);
        }
    }

    // Copy result back to global memory (DDR)
    loop_write_to_out_ddr:
    for (int i = 0; i < num_rows; i++) {
        #pragma HLS PIPELINE II=1
        out[i] = out_bram[i];
    }
}

} // extern "C"
