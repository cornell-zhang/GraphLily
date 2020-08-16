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

void crossbar(VECTOR_T vector_uram_one_channel[VECTOR_BUFFER_LEN/NUM_BANK_PER_HBM_CHANNEL + 1][NUM_BANK_PER_HBM_CHANNEL],
              unsigned int in_address[NUM_PE_PER_HBM_CHANNEL],
              bool in_valid[NUM_PE_PER_HBM_CHANNEL],
              VECTOR_T out_data[NUM_PE_PER_HBM_CHANNEL],
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

    assert(rotate >= 0);
    assert(rotate < NUM_PE_PER_HBM_CHANNEL);

    array_shift_left<bool>(in_valid, rotate);
    array_shift_left<unsigned int>(in_address, rotate);

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        in_bank_idx[PE_idx] = get_bank_idx(in_address[PE_idx]);
    }

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
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

    array_shift_right<unsigned int>(out_data, rotate);
}


static void read_data_one_channel(const PACKED_INDEX_T *indices_one_channel,
                                  hls::stream<PACKED_INDEX_T> &indices_stream_one_channel,
                                  unsigned int start,
                                  unsigned int end) {
    // Pass size to compute_spmv_one_channel by indices_stream_one_channel
    unsigned int size = end - start;
    PACKED_INDEX_T tmp;
    tmp.data[0] = size;
    indices_stream_one_channel << tmp;

    // Burst read
    loop_read_data_one_channel:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        indices_stream_one_channel << indices_one_channel[i + start];
    }
}


static void unpack_data_one_channel(hls::stream<PACKED_INDEX_T> &indices_stream_one_channel,
                                    hls::stream<unsigned int> indices_stream_one_PE[NUM_PE_PER_HBM_CHANNEL]) {
    unsigned int size = indices_stream_one_channel.read().data[0];
    for (unsigned int j = 0; j < NUM_PE_PER_HBM_CHANNEL; j++) {
        #pragma HLS UNROLL
        indices_stream_one_PE[j] << size;
    }

    PACKED_INDEX_T tmp;
    loop_unpack_data_one_channel:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        tmp = indices_stream_one_channel.read();
        for (int j = 0; j < NUM_PE_PER_HBM_CHANNEL; j++) {
            #pragma HLS UNROLL
            indices_stream_one_PE[j] << tmp.data[j];
        }
    }
}


static void read_vector_ddr_to_uram(const PACKED_VECTOR_T *vector,
                                    VECTOR_T vector_uram[NUM_HBM_CHANNEL][VECTOR_BUFFER_LEN/NUM_BANK_PER_HBM_CHANNEL + 1][NUM_BANK_PER_HBM_CHANNEL],
                                    unsigned int num_cols,
                                    unsigned int num_col_partitions,
                                    unsigned int col_partition_idx) {
    unsigned int size = VECTOR_BUFFER_LEN;
    if (col_partition_idx == (num_col_partitions - 1)) {
        size = num_cols - (num_col_partitions - 1) * VECTOR_BUFFER_LEN;
    }
    assert (size % VECTOR_PACK_SIZE == 0);
    unsigned int vsize = size / VECTOR_PACK_SIZE;
    PACKED_VECTOR_T tmp;

    loop_read_vector_ddr_to_uram:
    for (int i = 0; i < vsize; i++) {
        #pragma HLS PIPELINE II=1
        tmp = vector[col_partition_idx * VECTOR_BUFFER_LEN / VECTOR_PACK_SIZE + i];
        for (int j = 0; j < NUM_HBM_CHANNEL; j++) {
            #pragma HLS UNROLL
            for (int k = 0; k < VECTOR_PACK_SIZE; k++) {
                #pragma HLS UNROLL
                vector_uram[j][i * NUM_PORT_PER_BANK + k / NUM_BANK_PER_HBM_CHANNEL][k % NUM_BANK_PER_HBM_CHANNEL] = tmp.data[k];
            }
        }
    }
}


static void compute_spmv_one_channel(hls::stream<unsigned int> indices_stream_one_PE[NUM_PE_PER_HBM_CHANNEL],
                                     VECTOR_T vector_uram_one_channel[VECTOR_BUFFER_LEN/NUM_BANK_PER_HBM_CHANNEL + 1][NUM_BANK_PER_HBM_CHANNEL],
                                     hls::stream<VECTOR_T> out_stream_one_channel[NUM_PE_PER_HBM_CHANNEL]) {
    unsigned int size[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=size complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        size[PE_idx] = indices_stream_one_PE[PE_idx].read();
    }

    unsigned int index[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=index complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        index[PE_idx] = indices_stream_one_PE[PE_idx].read();
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

    unsigned int vector_data[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=vector_data complete

    bool out_valid[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=out_valid complete

    VECTOR_T tmp_out[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=tmp_out complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        tmp_out[PE_idx] = 0;
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
            if (index[PE_idx] == IDX_MARKER) {
                write[PE_idx] = true;
            } else {
                write[PE_idx] = false;
            }
            if (out_valid[PE_idx]) {
                accumulate[PE_idx] = true;
            } else {
                accumulate[PE_idx] = false;
            }
        }

        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            // consume a value; decrement size
            if (out_valid[PE_idx] || (index[PE_idx] == IDX_MARKER)) {
                size[PE_idx]--;
            }
        }

        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            if (size[PE_idx] == 0) {
                finished[PE_idx] = 1;
            }
        }

        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            if (finished[PE_idx]) {
                index[PE_idx] = 0; // a dummy number
                in_valid[PE_idx] = false;
            } else if (out_valid[PE_idx] || (index[PE_idx] == IDX_MARKER)) {
                if (indices_stream_one_PE[PE_idx].read_nb(index[PE_idx])) {
                    in_valid[PE_idx] = (index[PE_idx] != IDX_MARKER);
                } else {
                    in_valid[PE_idx] = false;
                }
            } else {
                index[PE_idx] = index[PE_idx];
                in_valid[PE_idx] = true;
            }
        }

        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            if (write[PE_idx]) {
                out_stream_one_channel[PE_idx] << tmp_out[PE_idx];
                tmp_out[PE_idx] = 0;
            } else if (accumulate[PE_idx]) {
#if defined(MulAddSemiring)
                tmp_out[PE_idx] = tmp_out[PE_idx] + vector_data[PE_idx];
#elif defined(LogicalAndOrSemiring)
                tmp_out[PE_idx] = tmp_out[PE_idx] || vector_data[PE_idx];
#else
                std::cout << "Invalid semiring" << std::endl;
                exit(EXIT_FAILURE);
#endif
            }
        }
    }
}


static void write_out_bram_one_PE(hls::stream<VECTOR_T> &s,
                                  VECTOR_T out_bram_one_PE[OUT_BUFFER_LEN/NUM_PE_TOTAL + 1],
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
    const PACKED_VECTOR_T *vector,
    const unsigned int *channel_0_partition_indptr,
    const PACKED_INDEX_T *channel_0_indices,
    const unsigned int *channel_1_partition_indptr,
    const PACKED_INDEX_T *channel_1_indices,
    const unsigned int *channel_2_partition_indptr,
    const PACKED_INDEX_T *channel_2_indices,
    const unsigned int *channel_3_partition_indptr,
    const PACKED_INDEX_T *channel_3_indices,
    const unsigned int *channel_4_partition_indptr,
    const PACKED_INDEX_T *channel_4_indices,
    const unsigned int *channel_5_partition_indptr,
    const PACKED_INDEX_T *channel_5_indices,
    const unsigned int *channel_6_partition_indptr,
    const PACKED_INDEX_T *channel_6_indices,
    const unsigned int *channel_7_partition_indptr,
    const PACKED_INDEX_T *channel_7_indices,
    const unsigned int *channel_8_partition_indptr,
    const PACKED_INDEX_T *channel_8_indices,
    const unsigned int *channel_9_partition_indptr,
    const PACKED_INDEX_T *channel_9_indices,
    const unsigned int *channel_10_partition_indptr,
    const PACKED_INDEX_T *channel_10_indices,
    const unsigned int *channel_11_partition_indptr,
    const PACKED_INDEX_T *channel_11_indices,
    const unsigned int *channel_12_partition_indptr,
    const PACKED_INDEX_T *channel_12_indices,
    const unsigned int *channel_13_partition_indptr,
    const PACKED_INDEX_T *channel_13_indices,
    const unsigned int *channel_14_partition_indptr,
    const PACKED_INDEX_T *channel_14_indices,
    const unsigned int *channel_15_partition_indptr,
    const PACKED_INDEX_T *channel_15_indices,
    // const unsigned int *channel_16_partition_indptr,
    // const PACKED_INDEX_T *channel_16_indices,
    // const unsigned int *channel_17_partition_indptr,
    // const PACKED_INDEX_T *channel_17_indices,
    // const unsigned int *channel_18_partition_indptr,
    // const PACKED_INDEX_T *channel_18_indices,
    // const unsigned int *channel_19_partition_indptr,
    // const PACKED_INDEX_T *channel_19_indices,
    // const unsigned int *channel_20_partition_indptr,
    // const PACKED_INDEX_T *channel_20_indices,
    // const unsigned int *channel_21_partition_indptr,
    // const PACKED_INDEX_T *channel_21_indices,
    // const unsigned int *channel_22_partition_indptr,
    // const PACKED_INDEX_T *channel_22_indices,
    // const unsigned int *channel_23_partition_indptr,
    // const PACKED_INDEX_T *channel_23_indices,
    // const unsigned int *channel_24_partition_indptr,
    // const PACKED_INDEX_T *channel_24_indices,
    // const unsigned int *channel_25_partition_indptr,
    // const PACKED_INDEX_T *channel_25_indices,
    // const unsigned int *channel_26_partition_indptr,
    // const PACKED_INDEX_T *channel_26_indices,
    // const unsigned int *channel_27_partition_indptr,
    // const PACKED_INDEX_T *channel_27_indices,
    // const unsigned int *channel_28_partition_indptr,
    // const PACKED_INDEX_T *channel_28_indices,
    // const unsigned int *channel_29_partition_indptr,
    // const PACKED_INDEX_T *channel_29_indices,
    // const unsigned int *channel_30_partition_indptr,
    // const PACKED_INDEX_T *channel_30_indices,
    // const unsigned int *channel_31_partition_indptr,
    // const PACKED_INDEX_T *channel_31_indices,
#if defined(USE_MASK)
    const PACKED_VECTOR_T *mask,
#endif
    PACKED_VECTOR_T *out,
    unsigned int num_rows,
    unsigned int num_cols
) {
#pragma HLS INTERFACE m_axi port=channel_0_indices offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=channel_1_indices offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=channel_2_indices offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=channel_3_indices offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=channel_4_indices offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=channel_5_indices offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=channel_6_indices offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=channel_7_indices offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=channel_8_indices offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=channel_9_indices offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=channel_10_indices offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=channel_11_indices offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=channel_12_indices offset=slave bundle=gmem12
#pragma HLS INTERFACE m_axi port=channel_13_indices offset=slave bundle=gmem13
#pragma HLS INTERFACE m_axi port=channel_14_indices offset=slave bundle=gmem14
#pragma HLS INTERFACE m_axi port=channel_15_indices offset=slave bundle=gmem15
// #pragma HLS INTERFACE m_axi port=channel_16_indices offset=slave bundle=gmem16
// #pragma HLS INTERFACE m_axi port=channel_17_indices offset=slave bundle=gmem17
// #pragma HLS INTERFACE m_axi port=channel_18_indices offset=slave bundle=gmem18
// #pragma HLS INTERFACE m_axi port=channel_19_indices offset=slave bundle=gmem19
// #pragma HLS INTERFACE m_axi port=channel_20_indices offset=slave bundle=gmem20
// #pragma HLS INTERFACE m_axi port=channel_21_indices offset=slave bundle=gmem21
// #pragma HLS INTERFACE m_axi port=channel_22_indices offset=slave bundle=gmem22
// #pragma HLS INTERFACE m_axi port=channel_23_indices offset=slave bundle=gmem23
// #pragma HLS INTERFACE m_axi port=channel_24_indices offset=slave bundle=gmem24
// #pragma HLS INTERFACE m_axi port=channel_25_indices offset=slave bundle=gmem25
// #pragma HLS INTERFACE m_axi port=channel_26_indices offset=slave bundle=gmem26
// #pragma HLS INTERFACE m_axi port=channel_27_indices offset=slave bundle=gmem27
// #pragma HLS INTERFACE m_axi port=channel_28_indices offset=slave bundle=gmem28
// #pragma HLS INTERFACE m_axi port=channel_29_indices offset=slave bundle=gmem29
// #pragma HLS INTERFACE m_axi port=channel_30_indices offset=slave bundle=gmem30
// #pragma HLS INTERFACE m_axi port=channel_31_indices offset=slave bundle=gmem31

#pragma HLS INTERFACE m_axi port=channel_0_partition_indptr offset=slave bundle=gmem32
#pragma HLS INTERFACE m_axi port=channel_1_partition_indptr offset=slave bundle=gmem33
#pragma HLS INTERFACE m_axi port=channel_2_partition_indptr offset=slave bundle=gmem34
#pragma HLS INTERFACE m_axi port=channel_3_partition_indptr offset=slave bundle=gmem35
#pragma HLS INTERFACE m_axi port=channel_4_partition_indptr offset=slave bundle=gmem36
#pragma HLS INTERFACE m_axi port=channel_5_partition_indptr offset=slave bundle=gmem37
#pragma HLS INTERFACE m_axi port=channel_6_partition_indptr offset=slave bundle=gmem38
#pragma HLS INTERFACE m_axi port=channel_7_partition_indptr offset=slave bundle=gmem39
#pragma HLS INTERFACE m_axi port=channel_8_partition_indptr offset=slave bundle=gmem40
#pragma HLS INTERFACE m_axi port=channel_9_partition_indptr offset=slave bundle=gmem41
#pragma HLS INTERFACE m_axi port=channel_10_partition_indptr offset=slave bundle=gmem42
#pragma HLS INTERFACE m_axi port=channel_11_partition_indptr offset=slave bundle=gmem43
#pragma HLS INTERFACE m_axi port=channel_12_partition_indptr offset=slave bundle=gmem44
#pragma HLS INTERFACE m_axi port=channel_13_partition_indptr offset=slave bundle=gmem45
#pragma HLS INTERFACE m_axi port=channel_14_partition_indptr offset=slave bundle=gmem46
#pragma HLS INTERFACE m_axi port=channel_15_partition_indptr offset=slave bundle=gmem47
// #pragma HLS INTERFACE m_axi port=channel_16_partition_indptr offset=slave bundle=gmem48
// #pragma HLS INTERFACE m_axi port=channel_17_partition_indptr offset=slave bundle=gmem49
// #pragma HLS INTERFACE m_axi port=channel_18_partition_indptr offset=slave bundle=gmem50
// #pragma HLS INTERFACE m_axi port=channel_19_partition_indptr offset=slave bundle=gmem51
// #pragma HLS INTERFACE m_axi port=channel_20_partition_indptr offset=slave bundle=gmem52
// #pragma HLS INTERFACE m_axi port=channel_21_partition_indptr offset=slave bundle=gmem53
// #pragma HLS INTERFACE m_axi port=channel_22_partition_indptr offset=slave bundle=gmem54
// #pragma HLS INTERFACE m_axi port=channel_23_partition_indptr offset=slave bundle=gmem55
// #pragma HLS INTERFACE m_axi port=channel_24_partition_indptr offset=slave bundle=gmem56
// #pragma HLS INTERFACE m_axi port=channel_25_partition_indptr offset=slave bundle=gmem57
// #pragma HLS INTERFACE m_axi port=channel_26_partition_indptr offset=slave bundle=gmem58
// #pragma HLS INTERFACE m_axi port=channel_27_partition_indptr offset=slave bundle=gmem59
// #pragma HLS INTERFACE m_axi port=channel_28_partition_indptr offset=slave bundle=gmem60
// #pragma HLS INTERFACE m_axi port=channel_29_partition_indptr offset=slave bundle=gmem61
// #pragma HLS INTERFACE m_axi port=channel_30_partition_indptr offset=slave bundle=gmem62
// #pragma HLS INTERFACE m_axi port=channel_31_partition_indptr offset=slave bundle=gmem63

#pragma HLS INTERFACE m_axi port=vector offset=slave bundle=gmem64
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem65
#if defined(USE_MASK)
#pragma HLS INTERFACE m_axi port=mask offset=slave bundle=gmem66
#endif

#pragma HLS INTERFACE s_axilite port=channel_0_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_1_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_2_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_3_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_4_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_5_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_6_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_7_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_8_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_9_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_10_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_11_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_12_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_13_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_14_indices bundle=control
#pragma HLS INTERFACE s_axilite port=channel_15_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_16_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_17_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_18_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_19_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_20_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_21_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_22_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_23_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_24_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_25_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_26_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_27_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_28_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_29_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_30_indices bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_31_indices bundle=control

#pragma HLS INTERFACE s_axilite port=channel_0_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_1_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_2_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_3_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_4_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_5_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_6_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_7_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_8_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_9_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_10_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_11_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_12_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_13_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_14_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_15_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_16_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_17_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_18_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_19_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_20_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_21_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_22_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_23_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_24_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_25_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_26_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_27_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_28_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_29_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_30_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_31_partition_indptr bundle=control

#pragma HLS INTERFACE s_axilite port=vector bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#if defined(USE_MASK)
#pragma HLS INTERFACE s_axilite port=mask bundle=control
#endif

#pragma HLS INTERFACE s_axilite port=num_rows bundle=control
#pragma HLS INTERFACE s_axilite port=num_cols bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATA_PACK variable=channel_0_indices
#pragma HLS DATA_PACK variable=channel_1_indices
#pragma HLS DATA_PACK variable=channel_2_indices
#pragma HLS DATA_PACK variable=channel_3_indices
#pragma HLS DATA_PACK variable=channel_4_indices
#pragma HLS DATA_PACK variable=channel_5_indices
#pragma HLS DATA_PACK variable=channel_6_indices
#pragma HLS DATA_PACK variable=channel_7_indices
#pragma HLS DATA_PACK variable=channel_8_indices
#pragma HLS DATA_PACK variable=channel_9_indices
#pragma HLS DATA_PACK variable=channel_10_indices
#pragma HLS DATA_PACK variable=channel_11_indices
#pragma HLS DATA_PACK variable=channel_12_indices
#pragma HLS DATA_PACK variable=channel_13_indices
#pragma HLS DATA_PACK variable=channel_14_indices
#pragma HLS DATA_PACK variable=channel_15_indices
// #pragma HLS DATA_PACK variable=channel_16_indices
// #pragma HLS DATA_PACK variable=channel_17_indices
// #pragma HLS DATA_PACK variable=channel_18_indices
// #pragma HLS DATA_PACK variable=channel_19_indices
// #pragma HLS DATA_PACK variable=channel_20_indices
// #pragma HLS DATA_PACK variable=channel_21_indices
// #pragma HLS DATA_PACK variable=channel_22_indices
// #pragma HLS DATA_PACK variable=channel_23_indices
// #pragma HLS DATA_PACK variable=channel_24_indices
// #pragma HLS DATA_PACK variable=channel_25_indices
// #pragma HLS DATA_PACK variable=channel_26_indices
// #pragma HLS DATA_PACK variable=channel_27_indices
// #pragma HLS DATA_PACK variable=channel_28_indices
// #pragma HLS DATA_PACK variable=channel_29_indices
// #pragma HLS DATA_PACK variable=channel_30_indices
// #pragma HLS DATA_PACK variable=channel_31_indices

#pragma HLS DATA_PACK variable=vector
#pragma HLS DATA_PACK variable=out
#if defined(USE_MASK)
#pragma HLS DATA_PACK variable=mask
#endif

    // All PEs within the same channel share one vector.
    VECTOR_T vector_uram[NUM_HBM_CHANNEL][VECTOR_BUFFER_LEN/NUM_BANK_PER_HBM_CHANNEL + 1][NUM_BANK_PER_HBM_CHANNEL];
    #pragma HLS RESOURCE variable=vector_uram core=XPM_MEMORY uram
    #pragma HLS ARRAY_PARTITION variable=vector_uram complete dim=1
    #pragma HLS ARRAY_PARTITION variable=vector_uram complete dim=3

    // There is no conflict when multiple PEs write to out_bram
    VECTOR_T out_bram[NUM_PE_TOTAL][OUT_BUFFER_LEN/NUM_PE_TOTAL + 1];
    #pragma HLS ARRAY_PARTITION variable=out_bram complete dim=1

    hls::stream<PACKED_INDEX_T> indices_stream[NUM_HBM_CHANNEL];
    /* Depth is set to the same for all the streams in one array */
    #pragma HLS STREAM variable=indices_stream depth=256

    hls::stream<unsigned int> unpacked_indices_stream[NUM_HBM_CHANNEL][NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS STREAM variable=unpacked_indices_stream depth=256

    hls::stream<VECTOR_T> vector_stream[NUM_HBM_CHANNEL][NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS STREAM variable=vector_stream depth=256

    hls::stream<VECTOR_T> out_stream[NUM_HBM_CHANNEL][NUM_PE_PER_HBM_CHANNEL];
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
            read_data_one_channel(channel_0_indices,
                                  indices_stream[0],
                                  channel_0_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_0_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_1_indices,
                                  indices_stream[1],
                                  channel_1_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_1_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_2_indices,
                                  indices_stream[2],
                                  channel_2_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_2_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_3_indices,
                                  indices_stream[3],
                                  channel_3_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_3_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_4_indices,
                                  indices_stream[4],
                                  channel_4_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_4_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_5_indices,
                                  indices_stream[5],
                                  channel_5_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_5_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_6_indices,
                                  indices_stream[6],
                                  channel_6_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_6_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_7_indices,
                                  indices_stream[7],
                                  channel_7_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_7_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_8_indices,
                                  indices_stream[8],
                                  channel_8_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_8_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_9_indices,
                                  indices_stream[9],
                                  channel_9_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_9_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_10_indices,
                                  indices_stream[10],
                                  channel_10_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_10_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_11_indices,
                                  indices_stream[11],
                                  channel_11_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_11_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_12_indices,
                                  indices_stream[12],
                                  channel_12_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_12_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_13_indices,
                                  indices_stream[13],
                                  channel_13_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_13_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_14_indices,
                                  indices_stream[14],
                                  channel_14_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_14_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            read_data_one_channel(channel_15_indices,
                                  indices_stream[15],
                                  channel_15_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
                                  channel_15_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_16_indices,
            //                       indices_stream[16],
            //                       channel_16_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_16_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_17_indices,
            //                       indices_stream[17],
            //                       channel_17_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_17_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_18_indices,
            //                       indices_stream[18],
            //                       channel_18_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_18_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_19_indices,
            //                       indices_stream[19],
            //                       channel_19_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_19_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_20_indices,
            //                       indices_stream[20],
            //                       channel_20_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_20_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_21_indices,
            //                       indices_stream[21],
            //                       channel_21_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_21_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_22_indices,
            //                       indices_stream[22],
            //                       channel_22_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_22_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_23_indices,
            //                       indices_stream[23],
            //                       channel_23_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_23_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_24_indices,
            //                       indices_stream[24],
            //                       channel_24_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_24_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_25_indices,
            //                       indices_stream[25],
            //                       channel_25_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_25_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_26_indices,
            //                       indices_stream[26],
            //                       channel_26_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_26_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_27_indices,
            //                       indices_stream[27],
            //                       channel_27_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_27_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_28_indices,
            //                       indices_stream[28],
            //                       channel_28_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_28_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_29_indices,
            //                       indices_stream[29],
            //                       channel_29_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_29_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_30_indices,
            //                       indices_stream[30],
            //                       channel_30_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_30_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            // read_data_one_channel(channel_31_indices,
            //                       indices_stream[31],
            //                       channel_31_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx],
            //                       channel_31_partition_indptr[row_partition_idx*num_col_partitions + col_partition_idx + 1]);
            unpack_data_one_channel(indices_stream[0], unpacked_indices_stream[0]);
            unpack_data_one_channel(indices_stream[1], unpacked_indices_stream[1]);
            unpack_data_one_channel(indices_stream[2], unpacked_indices_stream[2]);
            unpack_data_one_channel(indices_stream[3], unpacked_indices_stream[3]);
            unpack_data_one_channel(indices_stream[4], unpacked_indices_stream[4]);
            unpack_data_one_channel(indices_stream[5], unpacked_indices_stream[5]);
            unpack_data_one_channel(indices_stream[6], unpacked_indices_stream[6]);
            unpack_data_one_channel(indices_stream[7], unpacked_indices_stream[7]);
            unpack_data_one_channel(indices_stream[8], unpacked_indices_stream[8]);
            unpack_data_one_channel(indices_stream[9], unpacked_indices_stream[9]);
            unpack_data_one_channel(indices_stream[10], unpacked_indices_stream[10]);
            unpack_data_one_channel(indices_stream[11], unpacked_indices_stream[11]);
            unpack_data_one_channel(indices_stream[12], unpacked_indices_stream[12]);
            unpack_data_one_channel(indices_stream[13], unpacked_indices_stream[13]);
            unpack_data_one_channel(indices_stream[14], unpacked_indices_stream[14]);
            unpack_data_one_channel(indices_stream[15], unpacked_indices_stream[15]);
            // unpack_data_one_channel(indices_stream[16], unpacked_indices_stream[16]);
            // unpack_data_one_channel(indices_stream[17], unpacked_indices_stream[17]);
            // unpack_data_one_channel(indices_stream[18], unpacked_indices_stream[18]);
            // unpack_data_one_channel(indices_stream[19], unpacked_indices_stream[19]);
            // unpack_data_one_channel(indices_stream[20], unpacked_indices_stream[20]);
            // unpack_data_one_channel(indices_stream[21], unpacked_indices_stream[21]);
            // unpack_data_one_channel(indices_stream[22], unpacked_indices_stream[22]);
            // unpack_data_one_channel(indices_stream[23], unpacked_indices_stream[23]);
            // unpack_data_one_channel(indices_stream[24], unpacked_indices_stream[24]);
            // unpack_data_one_channel(indices_stream[25], unpacked_indices_stream[25]);
            // unpack_data_one_channel(indices_stream[26], unpacked_indices_stream[26]);
            // unpack_data_one_channel(indices_stream[27], unpacked_indices_stream[27]);
            // unpack_data_one_channel(indices_stream[28], unpacked_indices_stream[28]);
            // unpack_data_one_channel(indices_stream[29], unpacked_indices_stream[29]);
            // unpack_data_one_channel(indices_stream[30], unpacked_indices_stream[30]);
            // unpack_data_one_channel(indices_stream[31], unpacked_indices_stream[31]);

            compute_spmv_one_channel(unpacked_indices_stream[0],
                                     vector_uram[0],
                                     out_stream[0]);
            compute_spmv_one_channel(unpacked_indices_stream[1],
                                     vector_uram[1],
                                     out_stream[1]);
            compute_spmv_one_channel(unpacked_indices_stream[2],
                                     vector_uram[2],
                                     out_stream[2]);
            compute_spmv_one_channel(unpacked_indices_stream[3],
                                     vector_uram[3],
                                     out_stream[3]);
            compute_spmv_one_channel(unpacked_indices_stream[4],
                                     vector_uram[4],
                                     out_stream[4]);
            compute_spmv_one_channel(unpacked_indices_stream[5],
                                     vector_uram[5],
                                     out_stream[5]);
            compute_spmv_one_channel(unpacked_indices_stream[6],
                                     vector_uram[6],
                                     out_stream[6]);
            compute_spmv_one_channel(unpacked_indices_stream[7],
                                     vector_uram[7],
                                     out_stream[7]);
            compute_spmv_one_channel(unpacked_indices_stream[8],
                                     vector_uram[8],
                                     out_stream[8]);
            compute_spmv_one_channel(unpacked_indices_stream[9],
                                     vector_uram[9],
                                     out_stream[9]);
            compute_spmv_one_channel(unpacked_indices_stream[10],
                                     vector_uram[10],
                                     out_stream[10]);
            compute_spmv_one_channel(unpacked_indices_stream[11],
                                     vector_uram[11],
                                     out_stream[11]);
            compute_spmv_one_channel(unpacked_indices_stream[12],
                                     vector_uram[12],
                                     out_stream[12]);
            compute_spmv_one_channel(unpacked_indices_stream[13],
                                     vector_uram[13],
                                     out_stream[13]);
            compute_spmv_one_channel(unpacked_indices_stream[14],
                                     vector_uram[14],
                                     out_stream[14]);
            compute_spmv_one_channel(unpacked_indices_stream[15],
                                     vector_uram[15],
                                     out_stream[15]);
            // compute_spmv_one_channel(unpacked_indices_stream[16],
            //                          vector_uram[16],
            //                          out_stream[16]);
            // compute_spmv_one_channel(unpacked_indices_stream[17],
            //                          vector_uram[17],
            //                          out_stream[17]);
            // compute_spmv_one_channel(unpacked_indices_stream[18],
            //                          vector_uram[18],
            //                          out_stream[18]);
            // compute_spmv_one_channel(unpacked_indices_stream[19],
            //                          vector_uram[19],
            //                          out_stream[19]);
            // compute_spmv_one_channel(unpacked_indices_stream[20],
            //                          vector_uram[20],
            //                          out_stream[20]);
            // compute_spmv_one_channel(unpacked_indices_stream[21],
            //                          vector_uram[21],
            //                          out_stream[21]);
            // compute_spmv_one_channel(unpacked_indices_stream[22],
            //                          vector_uram[22],
            //                          out_stream[22]);
            // compute_spmv_one_channel(unpacked_indices_stream[23],
            //                          vector_uram[23],
            //                          out_stream[23]);
            // compute_spmv_one_channel(unpacked_indices_stream[24],
            //                          vector_uram[24],
            //                          out_stream[24]);
            // compute_spmv_one_channel(unpacked_indices_stream[25],
            //                          vector_uram[25],
            //                          out_stream[25]);
            // compute_spmv_one_channel(unpacked_indices_stream[26],
            //                          vector_uram[26],
            //                          out_stream[26]);
            // compute_spmv_one_channel(unpacked_indices_stream[27],
            //                          vector_uram[27],
            //                          out_stream[27]);
            // compute_spmv_one_channel(unpacked_indices_stream[28],
            //                          vector_uram[28],
            //                          out_stream[28]);
            // compute_spmv_one_channel(unpacked_indices_stream[29],
            //                          vector_uram[29],
            //                          out_stream[29]);
            // compute_spmv_one_channel(unpacked_indices_stream[30],
            //                          vector_uram[30],
            //                          out_stream[30]);
            // compute_spmv_one_channel(unpacked_indices_stream[31],
            //                          vector_uram[31],
            //                          out_stream[31]);
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[0], out_bram, 0, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[1], out_bram, 1, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[2], out_bram, 2, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[3], out_bram, 3, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[4], out_bram, 4, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[5], out_bram, 5, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[6], out_bram, 6, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[7], out_bram, 7, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[8], out_bram, 8, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[9], out_bram, 9, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[10], out_bram, 10, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[11], out_bram, 11, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[12], out_bram, 12, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[13], out_bram, 13, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[14], out_bram, 14, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[15], out_bram, 15, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[16], out_bram, 16, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[17], out_bram, 17, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[18], out_bram, 18, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[19], out_bram, 19, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[20], out_bram, 20, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[21], out_bram, 21, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[22], out_bram, 22, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[23], out_bram, 23, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[24], out_bram, 24, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[25], out_bram, 25, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[26], out_bram, 26, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[27], out_bram, 27, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[28], out_bram, 28, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[29], out_bram, 29, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[30], out_bram, 30, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[31], out_bram, 31, size)
        }

        assert (size % VECTOR_PACK_SIZE == 0);
        unsigned int vsize = size / VECTOR_PACK_SIZE;
        PACKED_VECTOR_T tmp_out;
#if defined(USE_MASK)
        PACKED_VECTOR_T tmp_mask;
#endif

        loop_write_to_out_ddr:
        for (int i = 0; i < vsize; i++) {
            #pragma HLS PIPELINE II=1
#if defined(USE_MASK)
            tmp_mask = mask[i + row_partition_idx * OUT_BUFFER_LEN / VECTOR_PACK_SIZE];
#endif
            for (int k = 0; k < VECTOR_PACK_SIZE; k++) {
                #pragma HLS UNROLL
#if not defined(USE_MASK)
                tmp_out.data[k] = out_bram[i % NUM_HBM_CHANNEL * VECTOR_PACK_SIZE + k][i / NUM_HBM_CHANNEL];
#elif defined(MASK_WRITE_TO_ZERO)
                if (tmp_mask.data[k] == 0) {
                    tmp_out.data[k] = out_bram[i % NUM_HBM_CHANNEL * VECTOR_PACK_SIZE + k][i / NUM_HBM_CHANNEL];
                } else {
                    tmp_out.data[k] = 0;
                }
#elif defined(MASK_WRITE_TO_ONE)
                if (tmp_mask.data[k] == 0) {
                    tmp_out.data[k] = 0;
                } else {
                    tmp_out.data[k] = out_bram[i % NUM_HBM_CHANNEL * VECTOR_PACK_SIZE + k][i / NUM_HBM_CHANNEL];
                }
#else
                std::cout << "Invalid mask type" << std::endl;
                exit(EXIT_FAILURE);
#endif
            }
            out[i + row_partition_idx * OUT_BUFFER_LEN / VECTOR_PACK_SIZE] = tmp_out;
        }
    }
}

} // extern "C"
