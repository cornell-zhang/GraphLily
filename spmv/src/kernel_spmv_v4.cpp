#include "kernel_spmv_v4.h"
#include <hls_stream.h>
#include <assert.h>


static void read_data_one_channel(const packed_index_t *indices,
                                  hls::stream<packed_index_t> &indices_stream,
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
    }
}


static void read_vector(const packed_data_t *vector,
                        data_t channel_0_vector_one_partition_bram[NUM_PE_PER_HBM_CHANNEL][VECTOR_BUFFER_LEN],
                        data_t channel_1_vector_one_partition_bram[NUM_PE_PER_HBM_CHANNEL][VECTOR_BUFFER_LEN],
                        unsigned int num_cols,
                        unsigned int num_col_partitions,
                        unsigned int partition_idx) {
    unsigned int size = VECTOR_BUFFER_LEN;
    if (partition_idx == (num_col_partitions - 1))
        size = num_cols - (num_col_partitions - 1) * VECTOR_BUFFER_LEN;
    assert (size % VDATA_SIZE == 0);
    unsigned int vsize = size / VDATA_SIZE;
    packed_data_t tmp;

    loop_read_vector:
    for (int i = 0; i < vsize; i++) {
        #pragma HLS PIPELINE II=1
        tmp = vector[partition_idx * VECTOR_BUFFER_LEN / VDATA_SIZE + i];
        for (int j = 0; j < NUM_PE_PER_HBM_CHANNEL; j++) {
            #pragma HLS UNROLL
            for (int k = 0; k < VDATA_SIZE; k++) {
                #pragma HLS UNROLL
                channel_0_vector_one_partition_bram[j][i * VDATA_SIZE + k] = tmp.data[k];
                channel_1_vector_one_partition_bram[j][i * VDATA_SIZE + k] = tmp.data[k];
            }
        }
    }
}


static void compute_spmv_one_channel(hls::stream<packed_index_t> &indices_stream,
                                     data_t vector_one_partition_bram[NUM_PE_PER_HBM_CHANNEL][VECTOR_BUFFER_LEN],
                                     hls::stream<data_t> out_stream[NUM_PE_PER_HBM_CHANNEL],
                                     unsigned int channel_idx) {
    unsigned int index[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=index complete

    data_t tmp_out[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=tmp_out complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        tmp_out[PE_idx] = 0;
    }

    unsigned int size = indices_stream.read().data[0];

    loop_compute_spmv_one_channel:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1

        packed_index_t packed_indices = indices_stream.read();

        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL

            index[PE_idx] = packed_indices.data[PE_idx];

            if (index[PE_idx] == IDX_MARKER) {
                out_stream[PE_idx] << tmp_out[PE_idx];
                tmp_out[PE_idx] = 0;
            } else {
                tmp_out[PE_idx] += vector_one_partition_bram[PE_idx][index[PE_idx]];
            }
        }
    }
}


static void write_out_bram_one_PE(hls::stream<data_t> &s,
                                  data_t out_bram_one_PE[MAX_NUM_ROWS/NUM_PE_TOTAL + 1],
                                  unsigned int PE_idx,
                                  unsigned int num_rows) {
    loop_write_out_bram_one_PE:
    for (int row_idx = 0; row_idx < num_rows; row_idx+=NUM_PE_TOTAL) {
        #pragma HLS PIPELINE II=1
        if ((row_idx + PE_idx) < num_rows) {
            out_bram_one_PE[row_idx/NUM_PE_TOTAL] += s.read();
        }
    }
}


#define WRITE_OUT_BRAM_ONE_CHANNEL(streams, brams, channel_idx, num_rows) { \
    unsigned int start_PE_idx = channel_idx * NUM_PE_PER_HBM_CHANNEL; \
    write_out_bram_one_PE(streams[0], brams[start_PE_idx + 0], start_PE_idx + 0, num_rows); \
    write_out_bram_one_PE(streams[1], brams[start_PE_idx + 1], start_PE_idx + 1, num_rows); \
    write_out_bram_one_PE(streams[2], brams[start_PE_idx + 2], start_PE_idx + 2, num_rows); \
    write_out_bram_one_PE(streams[3], brams[start_PE_idx + 3], start_PE_idx + 3, num_rows); \
    write_out_bram_one_PE(streams[4], brams[start_PE_idx + 4], start_PE_idx + 4, num_rows); \
    write_out_bram_one_PE(streams[5], brams[start_PE_idx + 5], start_PE_idx + 5, num_rows); \
    write_out_bram_one_PE(streams[6], brams[start_PE_idx + 6], start_PE_idx + 6, num_rows); \
    write_out_bram_one_PE(streams[7], brams[start_PE_idx + 7], start_PE_idx + 7, num_rows); \
    write_out_bram_one_PE(streams[8], brams[start_PE_idx + 8], start_PE_idx + 8, num_rows); \
    write_out_bram_one_PE(streams[9], brams[start_PE_idx + 9], start_PE_idx + 9, num_rows); \
    write_out_bram_one_PE(streams[10], brams[start_PE_idx + 10], start_PE_idx + 10, num_rows); \
    write_out_bram_one_PE(streams[11], brams[start_PE_idx + 11], start_PE_idx + 11, num_rows); \
    write_out_bram_one_PE(streams[12], brams[start_PE_idx + 12], start_PE_idx + 12, num_rows); \
    write_out_bram_one_PE(streams[13], brams[start_PE_idx + 13], start_PE_idx + 13, num_rows); \
    write_out_bram_one_PE(streams[14], brams[start_PE_idx + 14], start_PE_idx + 14, num_rows); \
    write_out_bram_one_PE(streams[15], brams[start_PE_idx + 15], start_PE_idx + 15, num_rows); \
} \


extern "C" {

void kernel_spmv_v4(
    const packed_data_t *vector,                     // The dense vector
    const unsigned int *channel_0_partition_indptr,  // Indptr of the partitions (the first channel)
    const packed_index_t *channel_0_indices,         // Indices (the first channel)
    const unsigned int *channel_1_partition_indptr,  // Indptr of the partitions (the second channel)
    const packed_index_t *channel_1_indices,         // Indices (the second channel)
    packed_data_t *out,                              // Output of the SpMV kernel
    unsigned int num_rows,                           // Number of rows of the sparse matrix
    unsigned int num_cols,                           // Number of columns of the sparse matrix
    unsigned int num_times                           // Running the kernel num_times for performance measurement
) {
#pragma HLS INTERFACE m_axi port=channel_0_indices          offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=channel_1_indices          offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=vector                     offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=channel_0_partition_indptr offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=channel_1_partition_indptr offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=out                        offset=slave bundle=gmem5

#pragma HLS INTERFACE s_axilite port=channel_0_indices          bundle=control
#pragma HLS INTERFACE s_axilite port=channel_1_indices          bundle=control
#pragma HLS INTERFACE s_axilite port=vector                     bundle=control
#pragma HLS INTERFACE s_axilite port=channel_0_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_1_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=out                        bundle=control

#pragma HLS INTERFACE s_axilite port=num_rows  bundle=control
#pragma HLS INTERFACE s_axilite port=num_cols  bundle=control
#pragma HLS INTERFACE s_axilite port=num_times bundle=control
#pragma HLS INTERFACE s_axilite port=return    bundle=control

#pragma HLS DATA_PACK variable=vector
#pragma HLS DATA_PACK variable=channel_0_indices
#pragma HLS DATA_PACK variable=channel_1_indices
#pragma HLS DATA_PACK variable=out

    data_t channel_0_vector_one_partition_bram[NUM_PE_PER_HBM_CHANNEL][VECTOR_BUFFER_LEN];
    #pragma HLS ARRAY_PARTITION variable=channel_0_vector_one_partition_bram complete dim=1
    #pragma HLS ARRAY_PARTITION variable=channel_0_vector_one_partition_bram cyclic factor=NUM_PE_PER_HBM_CHANNEL dim=2

    data_t channel_1_vector_one_partition_bram[NUM_PE_PER_HBM_CHANNEL][VECTOR_BUFFER_LEN];
    #pragma HLS ARRAY_PARTITION variable=channel_1_vector_one_partition_bram complete dim=1
    #pragma HLS ARRAY_PARTITION variable=channel_1_vector_one_partition_bram cyclic factor=NUM_PE_PER_HBM_CHANNEL dim=2

    data_t out_bram[NUM_PE_TOTAL][MAX_NUM_ROWS/NUM_PE_TOTAL + 1];
    #pragma HLS ARRAY_PARTITION variable=out_bram complete dim=1
    // #pragma HLS RESOURCE variable=out_bram core=XPM_MEMORY uram

    hls::stream<packed_index_t> channel_0_indices_stream;
    #pragma HLS STREAM variable=channel_0_indices_stream depth=512

    hls::stream<packed_index_t> channel_1_indices_stream;
    #pragma HLS STREAM variable=channel_1_indices_stream depth=512

    hls::stream<data_t> channel_0_out_stream[NUM_PE_PER_HBM_CHANNEL];
    hls::stream<data_t> channel_1_out_stream[NUM_PE_PER_HBM_CHANNEL];
    /* Depth is set to the same for all the streams in one array */
    #pragma HLS STREAM variable=channel_0_out_stream depth=32
    #pragma HLS STREAM variable=channel_1_out_stream depth=32

    unsigned int num_col_partitions = (num_cols + VECTOR_BUFFER_LEN - 1) / VECTOR_BUFFER_LEN;

    // Running the same kernel num_times for performance measurement
    for (int count = 0; count < num_times; count++) {

        loop_initialize_out_bram:
        for (int row_idx = 0; row_idx < num_rows; row_idx+=NUM_PE_TOTAL) {
            for (int PE_idx = 0; PE_idx < NUM_PE_TOTAL; PE_idx++) {
                #pragma HLS UNROLL
                out_bram[PE_idx][row_idx/NUM_PE_TOTAL] = 0;
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
                                  channel_0_indices_stream,
                                  channel_0_partition_indptr[partition_idx],
                                  channel_0_partition_indptr[partition_idx + 1]);
            read_data_one_channel(channel_1_indices,
                                  channel_1_indices_stream,
                                  channel_1_partition_indptr[partition_idx],
                                  channel_1_partition_indptr[partition_idx + 1]);
            compute_spmv_one_channel(channel_0_indices_stream,
                                     channel_0_vector_one_partition_bram,
                                     channel_0_out_stream,
                                     0);
            compute_spmv_one_channel(channel_1_indices_stream,
                                     channel_1_vector_one_partition_bram,
                                     channel_1_out_stream,
                                     1);
            WRITE_OUT_BRAM_ONE_CHANNEL(channel_0_out_stream, out_bram, 0, num_rows)
            WRITE_OUT_BRAM_ONE_CHANNEL(channel_1_out_stream, out_bram, 1, num_rows)
        }
    }

    assert (num_rows % VDATA_SIZE == 0);
    unsigned int vsize = num_rows / VDATA_SIZE;
    packed_data_t tmp;

    loop_write_to_out_ddr:
    for (int i = 0; i < vsize; i++) {
        #pragma HLS PIPELINE II=1
        for (int k = 0; k < VDATA_SIZE; k++) {
            #pragma HLS UNROLL
            tmp.data[k] = out_bram[i % NUM_HBM_CHANNEL * VDATA_SIZE + k][i / NUM_HBM_CHANNEL];
        }
        out[i] = tmp;
    }
}

} // extern "C"