#include <hls_stream.h>
#include <ap_fixed.h>
#include <assert.h>
#include <iostream>

#include "./kernel_spmv.h"


unsigned int log2(unsigned int x) {
    switch (x) {
        case    1: return 0;
        case    2: return 1;
        case    4: return 2;
        case    8: return 3;
        case   16: return 4;
        default  : return 0;
    }
}

bool bool_array_and(bool array[NUM_PE_PER_HBM_CHANNEL]) {
    // #pragma HLS INLINE off
    bool result = true;
    for (int i = 0; i < NUM_PE_PER_HBM_CHANNEL; i++) {
        #pragma HLS UNROLL
        result = result & array[i];
    }
    return result;
}

unsigned int unsigned_array_max(unsigned int array[NUM_PE_PER_HBM_CHANNEL]) {
    // #pragma HLS INLINE off
    unsigned int result = 0;
    for (int i = 0; i < NUM_PE_PER_HBM_CHANNEL; i++) {
        #pragma HLS UNROLL
        result = (array[i] > result)? array[i] : result;
    }
    return result;
}

// Cyclic partitioning
unsigned int get_bank_idx(unsigned int full_addr) {
    return full_addr & ((1 << log2(NUM_BANK_PER_HBM_CHANNEL)) - 1);
}

// Cyclic partitioning
unsigned int get_bank_address(unsigned int full_addr) {
    return full_addr >> log2(NUM_BANK_PER_HBM_CHANNEL);
}

template <typename T>
void array_shift_left(T array[NUM_PE_PER_HBM_CHANNEL], unsigned int rotate) {
    T array_swap[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=array_swap complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        array_swap[PE_idx] = array[(PE_idx + rotate) % NUM_PE_PER_HBM_CHANNEL];
    }
    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        array[PE_idx] = array_swap[PE_idx];
    }
}

template <typename T>
void array_shift_right(T array[NUM_PE_PER_HBM_CHANNEL], unsigned int rotate) {
    T array_swap[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=array_swap complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        array_swap[PE_idx] = array[(PE_idx + NUM_PE_PER_HBM_CHANNEL - rotate) % NUM_PE_PER_HBM_CHANNEL];
    }
    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        array[PE_idx] = array_swap[PE_idx];
    }
}

void crossbar(VAL_T vector_uram_one_channel[VECTOR_BUFFER_LEN/NUM_BANK_PER_HBM_CHANNEL + 1][NUM_BANK_PER_HBM_CHANNEL],
              INDEX_T in_address[NUM_PE_PER_HBM_CHANNEL],
              bool in_valid[NUM_PE_PER_HBM_CHANNEL],
              VAL_T out_data[NUM_PE_PER_HBM_CHANNEL],
              bool out_valid[NUM_PE_PER_HBM_CHANNEL],
              unsigned int rotate) {
    #pragma HLS INLINE

    unsigned int bank_idx_to_PE_idx[NUM_BANK_PER_HBM_CHANNEL][NUM_PORT_PER_BANK];
    #pragma HLS ARRAY_PARTITION variable=bank_idx_to_PE_idx complete dim=0

    unsigned int bank_address[NUM_BANK_PER_HBM_CHANNEL][NUM_PORT_PER_BANK];
    #pragma HLS ARRAY_PARTITION variable=bank_address complete dim=0

    unsigned int bank_num_valid_requests[NUM_BANK_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=bank_num_valid_requests complete

    for (int bank_idx = 0; bank_idx < NUM_BANK_PER_HBM_CHANNEL; bank_idx++) {
        #pragma HLS UNROLL
        bank_num_valid_requests[bank_idx] = 0;
    }

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        out_valid[PE_idx] = false;
    }

    unsigned int in_bank_idx[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=in_bank_idx complete dim=1

    unsigned int in_bank_address[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=in_bank_address complete dim=1

    assert(rotate < NUM_PE_PER_HBM_CHANNEL);

    array_shift_left<bool>(in_valid, rotate);
    array_shift_left<unsigned int>(in_address, rotate);

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        in_bank_idx[PE_idx] = get_bank_idx(in_address[PE_idx]);
    }

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        in_bank_address[PE_idx] = get_bank_address(in_address[PE_idx]);
    }

    loop_crossbar_bank_idx:
    for (int bank_idx = 0; bank_idx < NUM_BANK_PER_HBM_CHANNEL; bank_idx++) {
        #pragma HLS UNROLL
        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            if (in_valid[PE_idx] && (in_bank_idx[PE_idx] == bank_idx)) {
                unsigned int port_idx = bank_num_valid_requests[bank_idx];
                if (port_idx < NUM_PORT_PER_BANK) {
                    bank_idx_to_PE_idx[bank_idx][port_idx] = PE_idx;
                    bank_address[bank_idx][port_idx] = in_bank_address[PE_idx];
                    bank_num_valid_requests[bank_idx]++;
                    out_valid[PE_idx] = true;
                }
            }
        }
    }

    array_shift_right<bool>(in_valid, rotate);
    array_shift_right<unsigned int>(in_address, rotate);

    array_shift_right<bool>(out_valid, rotate);

    for (int bank_idx = 0; bank_idx < NUM_BANK_PER_HBM_CHANNEL; bank_idx++) {
        #pragma HLS UNROLL
        for (unsigned int port_idx = 0; port_idx < NUM_PORT_PER_BANK; port_idx++) {
            #pragma HLS UNROLL
            if (port_idx < bank_num_valid_requests[bank_idx]) {
                out_data[bank_idx_to_PE_idx[bank_idx][port_idx]] =
                    vector_uram_one_channel[bank_address[bank_idx][port_idx]][bank_idx];
            }
        }
    }

    array_shift_right<VAL_T>(out_data, rotate);
}


static void read_matrix_one_channel(const PACKET_T *matrix_one_channel,
                                    hls::stream<PACKED_INDEX_T> &indices_stream_one_channel,
                                    hls::stream<PACKED_VAL_T> &vals_stream_one_channel,
                                    const INDEX_T x[PACK_SIZE + 1]) {
    // Pass size to compute_spmv_one_channel by indices_stream_one_channel
    unsigned int start = x[0];
    PACKED_INDEX_T size;
    for (unsigned int k = 0; k < PACK_SIZE; k++) {
        #pragma HLS UNROLL
        size.data[k] = x[k+1] - 0;
    }
    indices_stream_one_channel << size;
    unsigned int max_size = unsigned_array_max(size.data);

    // Burst read
    PACKET_T packet;
    loop_read_matrix_one_channel:
    for (int i = 0; i < max_size; i++) {
        #pragma HLS PIPELINE II=1
        packet = matrix_one_channel[i + start];
        indices_stream_one_channel << packet.indices;
        vals_stream_one_channel << packet.vals;
    }
}


static void unpack_matrix_one_channel(hls::stream<PACKED_INDEX_T> &indices_stream_one_channel,
                                      hls::stream<INDEX_T> indices_stream_one_PE[NUM_PE_PER_HBM_CHANNEL],
                                      hls::stream<PACKED_VAL_T> &vals_stream_one_channel,
                                      hls::stream<VAL_T> vals_stream_one_PE[NUM_PE_PER_HBM_CHANNEL]) {
    PACKED_INDEX_T size = indices_stream_one_channel.read();
    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        indices_stream_one_PE[PE_idx] << size.data[PE_idx];
    }

    unsigned int max_size = unsigned_array_max(size.data);
    PACKED_INDEX_T tmp_index;
    PACKED_VAL_T tmp_val;

    loop_unpack_matrix_one_channel:
    for (int i = 0; i < max_size; i++) {
        #pragma HLS PIPELINE II=1
        tmp_index = indices_stream_one_channel.read();
        tmp_val = vals_stream_one_channel.read();
        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            if (i < size.data[PE_idx]) {
                indices_stream_one_PE[PE_idx] << tmp_index.data[PE_idx];
                vals_stream_one_PE[PE_idx] << tmp_val.data[PE_idx];
            }
        }
    }
}


static void read_vector_ddr_to_uram(const PACKED_VAL_T *vector,
                                    VAL_T vector_uram[NUM_HBM_CHANNEL][VECTOR_BUFFER_LEN/NUM_BANK_PER_HBM_CHANNEL + 1][NUM_BANK_PER_HBM_CHANNEL],
                                    unsigned int num_cols,
                                    unsigned int num_col_partitions,
                                    unsigned int col_partition_idx) {
    unsigned int size = VECTOR_BUFFER_LEN;
    if (col_partition_idx == (num_col_partitions - 1)) {
        size = num_cols - (num_col_partitions - 1) * VECTOR_BUFFER_LEN;
    }
    assert (size % PACK_SIZE == 0);
    unsigned int vsize = size / PACK_SIZE;
    PACKED_VAL_T tmp;

    loop_read_vector_ddr_to_uram:
    for (int i = 0; i < vsize; i++) {
        #pragma HLS PIPELINE II=1
        tmp = vector[col_partition_idx * VECTOR_BUFFER_LEN / PACK_SIZE + i];
        for (int j = 0; j < NUM_HBM_CHANNEL; j++) {
            #pragma HLS UNROLL
            for (int k = 0; k < PACK_SIZE; k++) {
                #pragma HLS UNROLL
                vector_uram[j][i * NUM_PORT_PER_BANK + k / NUM_BANK_PER_HBM_CHANNEL][k % NUM_BANK_PER_HBM_CHANNEL] = tmp.data[k];
            }
        }
    }
}


static void compute_spmv_one_channel(hls::stream<INDEX_T> indices_stream_one_PE[NUM_PE_PER_HBM_CHANNEL],
                                     hls::stream<VAL_T> vals_stream_one_PE[NUM_PE_PER_HBM_CHANNEL],
                                     VAL_T vector_uram_one_channel[VECTOR_BUFFER_LEN/NUM_BANK_PER_HBM_CHANNEL + 1][NUM_BANK_PER_HBM_CHANNEL],
                                     hls::stream<VAL_T> out_stream_one_channel[NUM_PE_PER_HBM_CHANNEL]) {
    int size[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=size complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        size[PE_idx] = indices_stream_one_PE[PE_idx].read();
    }

    INDEX_T index[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=index complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        index[PE_idx] = indices_stream_one_PE[PE_idx].read();
    }

    VAL_T val[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=val complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        val[PE_idx] = vals_stream_one_PE[PE_idx].read();
    }

    bool in_valid[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=in_valid complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        in_valid[PE_idx] = (index[PE_idx] != IDX_MARKER);
    }

    bool finished[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=finished complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        finished[PE_idx] = false;
    }

    VAL_T vector_data[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=vector_data complete

    bool out_valid[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=out_valid complete

    VAL_T tmp_out[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=tmp_out complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        tmp_out[PE_idx] = 0;
    }

    bool fifo_empty[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=fifo_empty complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        fifo_empty[PE_idx] = false;
    }

    bool write[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=write complete

    bool accumulate[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=accumulate complete

    unsigned int rotate = 0;

    loop_compute_spmv_one_channel:
    while (!bool_array_and(finished)) {
        #pragma HLS PIPELINE II=1

        crossbar(vector_uram_one_channel, index, in_valid, vector_data, out_valid, rotate);
        rotate++;
        if (rotate >= NUM_PE_PER_HBM_CHANNEL) rotate -= NUM_PE_PER_HBM_CHANNEL;

        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            if (!finished[PE_idx] && (index[PE_idx] == IDX_MARKER) && !fifo_empty[PE_idx]) {
                write[PE_idx] = true;
            } else {
                write[PE_idx] = false;
            }
            if (!finished[PE_idx] && out_valid[PE_idx] && !fifo_empty[PE_idx]) {
                accumulate[PE_idx] = true;
            } else {
                accumulate[PE_idx] = false;
            }
        }

        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            // consume a value; decrement size
            if ((out_valid[PE_idx] || (index[PE_idx] == IDX_MARKER)) && !fifo_empty[PE_idx]) {
                size[PE_idx]--;
            }
        }

        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            if (size[PE_idx] <= 0) {
                finished[PE_idx] = 1;
            }
        }

        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            if (finished[PE_idx]) {
                index[PE_idx] = 0; // a dummy number
                in_valid[PE_idx] = false;
            } else if (out_valid[PE_idx] || (index[PE_idx] == IDX_MARKER) || fifo_empty[PE_idx]) {
                if (indices_stream_one_PE[PE_idx].read_nb(index[PE_idx])) {
                    fifo_empty[PE_idx] = false;
                    in_valid[PE_idx] = (index[PE_idx] != IDX_MARKER);
                    val[PE_idx] = vals_stream_one_PE[PE_idx].read();
                } else {
                    fifo_empty[PE_idx] = true;
                    in_valid[PE_idx] = false;
                }
            } else {
                index[PE_idx] = index[PE_idx];
                in_valid[PE_idx] = true;
                val[PE_idx] = val[PE_idx];
            }
        }

        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            if (write[PE_idx]) {
                out_stream_one_channel[PE_idx] << tmp_out[PE_idx];
                tmp_out[PE_idx] = 0;
            } else if (accumulate[PE_idx]) {
#if defined(MulAddSemiring)
                tmp_out[PE_idx] = tmp_out[PE_idx] + val[PE_idx] * vector_data[PE_idx];
#elif defined(LogicalAndOrSemiring)
                tmp_out[PE_idx] = tmp_out[PE_idx] || (val[PE_idx] && vector_data[PE_idx]);
#else
                std::cout << "Invalid semiring" << std::endl;
                exit(EXIT_FAILURE);
#endif
            }
        }
    }
}


static void write_out_bram_one_PE(hls::stream<VAL_T> &s,
                                  VAL_T out_bram_one_PE[OUT_BUFFER_LEN/NUM_PE_TOTAL + 1],
                                  unsigned int PE_idx,
                                  unsigned int num_rows) {
    loop_write_out_bram_one_PE:
    for (int row_idx = 0; row_idx < num_rows; row_idx+=NUM_PE_TOTAL) {
        #pragma HLS PIPELINE II=1
        if ((row_idx + PE_idx) < num_rows) {
#if defined(MulAddSemiring)
            out_bram_one_PE[row_idx/NUM_PE_TOTAL] = s.read() + out_bram_one_PE[row_idx/NUM_PE_TOTAL];
#elif defined(LogicalAndOrSemiring)
            out_bram_one_PE[row_idx/NUM_PE_TOTAL] = s.read() || out_bram_one_PE[row_idx/NUM_PE_TOTAL];
#else
            std::cout << "Invalid semiring" << std::endl;
            exit(EXIT_FAILURE);
#endif
        }
    }

// #if not defined(__SYNTHESIS__)
//     unsigned int num_leftover = 0;
//     while (!s.empty()) {
//         num_leftover++;
//         s.read();
//     }
//     std::cout << "PE_idx: " << PE_idx << std::endl;
//     std::cout << "num_leftover: " << num_leftover << std::endl;
// #endif
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

void kernel_spmv(
    const PACKED_VAL_T *vector,
    const INDEX_T *channel_0_partition_indptr,
    const PACKET_T *channel_0_matrix,
    const INDEX_T *channel_1_partition_indptr,
    const PACKET_T *channel_1_matrix,
    const INDEX_T *channel_2_partition_indptr,
    const PACKET_T *channel_2_matrix,
    const INDEX_T *channel_3_partition_indptr,
    const PACKET_T *channel_3_matrix,
    const INDEX_T *channel_4_partition_indptr,
    const PACKET_T *channel_4_matrix,
    const INDEX_T *channel_5_partition_indptr,
    const PACKET_T *channel_5_matrix,
    const INDEX_T *channel_6_partition_indptr,
    const PACKET_T *channel_6_matrix,
    const INDEX_T *channel_7_partition_indptr,
    const PACKET_T *channel_7_matrix,
#if defined(USE_MASK)
    const PACKED_VAL_T *mask,
#endif
    PACKED_VAL_T *out,
    unsigned int num_rows,
    unsigned int num_cols
) {
#pragma HLS INTERFACE m_axi port=channel_0_matrix offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=channel_1_matrix offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=channel_2_matrix offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=channel_3_matrix offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=channel_4_matrix offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=channel_5_matrix offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=channel_6_matrix offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=channel_7_matrix offset=slave bundle=gmem7

#pragma HLS INTERFACE m_axi port=channel_0_partition_indptr offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=channel_1_partition_indptr offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=channel_2_partition_indptr offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=channel_3_partition_indptr offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=channel_4_partition_indptr offset=slave bundle=gmem12
#pragma HLS INTERFACE m_axi port=channel_5_partition_indptr offset=slave bundle=gmem13
#pragma HLS INTERFACE m_axi port=channel_6_partition_indptr offset=slave bundle=gmem14
#pragma HLS INTERFACE m_axi port=channel_7_partition_indptr offset=slave bundle=gmem15

#pragma HLS INTERFACE m_axi port=vector offset=slave bundle=gmem16
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem17
#if defined(USE_MASK)
#pragma HLS INTERFACE m_axi port=mask offset=slave bundle=gmem18
#endif

#pragma HLS INTERFACE s_axilite port=channel_0_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_1_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_2_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_3_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_4_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_5_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_6_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_7_matrix bundle=control

#pragma HLS INTERFACE s_axilite port=channel_0_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_1_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_2_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_3_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_4_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_5_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_6_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_7_partition_indptr bundle=control

#pragma HLS INTERFACE s_axilite port=vector bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#if defined(USE_MASK)
#pragma HLS INTERFACE s_axilite port=mask bundle=control
#endif

#pragma HLS INTERFACE s_axilite port=num_rows bundle=control
#pragma HLS INTERFACE s_axilite port=num_cols bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATA_PACK variable=channel_0_matrix
#pragma HLS DATA_PACK variable=channel_1_matrix
#pragma HLS DATA_PACK variable=channel_2_matrix
#pragma HLS DATA_PACK variable=channel_3_matrix
#pragma HLS DATA_PACK variable=channel_4_matrix
#pragma HLS DATA_PACK variable=channel_5_matrix
#pragma HLS DATA_PACK variable=channel_6_matrix
#pragma HLS DATA_PACK variable=channel_7_matrix

#pragma HLS DATA_PACK variable=vector
#pragma HLS DATA_PACK variable=out
#if defined(USE_MASK)
#pragma HLS DATA_PACK variable=mask
#endif

    // All PEs within the same channel share one vector.
    VAL_T vector_uram[NUM_HBM_CHANNEL][VECTOR_BUFFER_LEN/NUM_BANK_PER_HBM_CHANNEL + 1][NUM_BANK_PER_HBM_CHANNEL];
    #pragma HLS RESOURCE variable=vector_uram core=XPM_MEMORY uram
    #pragma HLS ARRAY_PARTITION variable=vector_uram complete dim=1
    #pragma HLS ARRAY_PARTITION variable=vector_uram complete dim=3

    // There is no conflict when multiple PEs write to out_bram
    VAL_T out_bram[NUM_PE_TOTAL][OUT_BUFFER_LEN/NUM_PE_TOTAL + 1];
    #pragma HLS ARRAY_PARTITION variable=out_bram complete dim=1

    hls::stream<PACKED_INDEX_T> indices_stream[NUM_HBM_CHANNEL];
    /* Depth is set to the same for all the streams in one array */
    #pragma HLS STREAM variable=indices_stream depth=512

    hls::stream<INDEX_T> unpacked_indices_stream[NUM_HBM_CHANNEL][NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS STREAM variable=unpacked_indices_stream depth=512

    hls::stream<PACKED_VAL_T> vals_stream[NUM_HBM_CHANNEL];
    #pragma HLS STREAM variable=vals_stream depth=512

    hls::stream<VAL_T> unpacked_vals_stream[NUM_HBM_CHANNEL][NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS STREAM variable=unpacked_vals_stream depth=512

    hls::stream<VAL_T> out_stream[NUM_HBM_CHANNEL][NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS STREAM variable=out_stream depth=32

    unsigned int num_row_partitions = (num_rows + OUT_BUFFER_LEN - 1) / OUT_BUFFER_LEN;
    unsigned int num_col_partitions = (num_cols + VECTOR_BUFFER_LEN - 1) / VECTOR_BUFFER_LEN;

    // Iterate row partitions
    for (int row_partition_idx = 0; row_partition_idx < num_row_partitions; row_partition_idx++) {

        unsigned int size = OUT_BUFFER_LEN;
        if (row_partition_idx == (num_row_partitions - 1)) {
            size = num_rows - (num_row_partitions - 1) * OUT_BUFFER_LEN;
        }

        loop_initialize_out_bram:
        for (int row_idx = 0; row_idx < size; row_idx+=NUM_PE_TOTAL) {
            for (int PE_idx = 0; PE_idx < NUM_PE_TOTAL; PE_idx++) {
                #pragma HLS UNROLL
                out_bram[PE_idx][row_idx/NUM_PE_TOTAL] = 0;
            }
        }

        // Iterate column partitions
        for (int col_partition_idx = 0; col_partition_idx < num_col_partitions; col_partition_idx++) {
            #pragma HLS dataflow

            read_vector_ddr_to_uram(vector,
                                    vector_uram,
                                    num_cols,
                                    num_col_partitions,
                                    col_partition_idx);

            read_matrix_one_channel(channel_0_matrix,
                                    indices_stream[0],
                                    vals_stream[0],
                                    &channel_0_partition_indptr[(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_1_matrix,
                                    indices_stream[1],
                                    vals_stream[1],
                                    &channel_1_partition_indptr[(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_2_matrix,
                                    indices_stream[2],
                                    vals_stream[2],
                                    &channel_2_partition_indptr[(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_3_matrix,
                                    indices_stream[3],
                                    vals_stream[3],
                                    &channel_3_partition_indptr[(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_4_matrix,
                                    indices_stream[4],
                                    vals_stream[4],
                                    &channel_4_partition_indptr[(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_5_matrix,
                                    indices_stream[5],
                                    vals_stream[5],
                                    &channel_5_partition_indptr[(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_6_matrix,
                                    indices_stream[6],
                                    vals_stream[6],
                                    &channel_6_partition_indptr[(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_7_matrix,
                                    indices_stream[7],
                                    vals_stream[7],
                                    &channel_7_partition_indptr[(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);

            unpack_matrix_one_channel(indices_stream[0],
                                      unpacked_indices_stream[0],
                                      vals_stream[0],
                                      unpacked_vals_stream[0]);
            unpack_matrix_one_channel(indices_stream[1],
                                      unpacked_indices_stream[1],
                                      vals_stream[1],
                                      unpacked_vals_stream[1]);
            unpack_matrix_one_channel(indices_stream[2],
                                      unpacked_indices_stream[2],
                                      vals_stream[2],
                                      unpacked_vals_stream[2]);
            unpack_matrix_one_channel(indices_stream[3],
                                      unpacked_indices_stream[3],
                                      vals_stream[3],
                                      unpacked_vals_stream[3]);
            unpack_matrix_one_channel(indices_stream[4],
                                      unpacked_indices_stream[4],
                                      vals_stream[4],
                                      unpacked_vals_stream[4]);
            unpack_matrix_one_channel(indices_stream[5],
                                      unpacked_indices_stream[5],
                                      vals_stream[5],
                                      unpacked_vals_stream[5]);
            unpack_matrix_one_channel(indices_stream[6],
                                      unpacked_indices_stream[6],
                                      vals_stream[6],
                                      unpacked_vals_stream[6]);
            unpack_matrix_one_channel(indices_stream[7],
                                      unpacked_indices_stream[7],
                                      vals_stream[7],
                                      unpacked_vals_stream[7]);

            compute_spmv_one_channel(unpacked_indices_stream[0],
                                     unpacked_vals_stream[0],
                                     vector_uram[0],
                                     out_stream[0]);
            compute_spmv_one_channel(unpacked_indices_stream[1],
                                     unpacked_vals_stream[1],
                                     vector_uram[1],
                                     out_stream[1]);
            compute_spmv_one_channel(unpacked_indices_stream[2],
                                     unpacked_vals_stream[2],
                                     vector_uram[2],
                                     out_stream[2]);
            compute_spmv_one_channel(unpacked_indices_stream[3],
                                     unpacked_vals_stream[3],
                                     vector_uram[3],
                                     out_stream[3]);
            compute_spmv_one_channel(unpacked_indices_stream[4],
                                     unpacked_vals_stream[4],
                                     vector_uram[4],
                                     out_stream[4]);
            compute_spmv_one_channel(unpacked_indices_stream[5],
                                     unpacked_vals_stream[5],
                                     vector_uram[5],
                                     out_stream[5]);
            compute_spmv_one_channel(unpacked_indices_stream[6],
                                     unpacked_vals_stream[6],
                                     vector_uram[6],
                                     out_stream[6]);
            compute_spmv_one_channel(unpacked_indices_stream[7],
                                     unpacked_vals_stream[7],
                                     vector_uram[7],
                                     out_stream[7]);

            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[0], out_bram, 0, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[1], out_bram, 1, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[2], out_bram, 2, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[3], out_bram, 3, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[4], out_bram, 4, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[5], out_bram, 5, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[6], out_bram, 6, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[7], out_bram, 7, size)
        }

        assert (size % PACK_SIZE == 0);
        unsigned int vsize = size / PACK_SIZE;
        PACKED_VAL_T tmp_out;
#if defined(USE_MASK)
        PACKED_VAL_T tmp_mask;
#endif

        loop_write_to_out_ddr:
        for (int i = 0; i < vsize; i++) {
            #pragma HLS PIPELINE II=1
#if defined(USE_MASK)
            tmp_mask = mask[i + row_partition_idx * OUT_BUFFER_LEN / PACK_SIZE];
#endif
            for (int k = 0; k < PACK_SIZE; k++) {
                #pragma HLS UNROLL
#if not defined(USE_MASK)
                tmp_out.data[k] = out_bram[i % NUM_HBM_CHANNEL * PACK_SIZE + k][i / NUM_HBM_CHANNEL];
#elif defined(MASK_WRITE_TO_ZERO)
                if (tmp_mask.data[k] == 0) {
                    tmp_out.data[k] = out_bram[i % NUM_HBM_CHANNEL * PACK_SIZE + k][i / NUM_HBM_CHANNEL];
                } else {
                    tmp_out.data[k] = 0;
                }
#elif defined(MASK_WRITE_TO_ONE)
                if (tmp_mask.data[k] == 0) {
                    tmp_out.data[k] = 0;
                } else {
                    tmp_out.data[k] = out_bram[i % NUM_HBM_CHANNEL * PACK_SIZE + k][i / NUM_HBM_CHANNEL];
                }
#else
                std::cout << "Invalid mask type" << std::endl;
                exit(EXIT_FAILURE);
#endif
            }
            out[i + row_partition_idx * OUT_BUFFER_LEN / PACK_SIZE] = tmp_out;
        }
    }
}

} // extern "C"
