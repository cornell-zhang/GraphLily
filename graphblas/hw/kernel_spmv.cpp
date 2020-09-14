#include <hls_stream.h>
#include <ap_fixed.h>
#include <assert.h>
#include <iostream>
#include <iomanip>

#include "./kernel_spmv.h"

//------------------------------------------------------------
// line tracing swtiches
//------------------------------------------------------------

#ifndef __SYNTHESIS__
static bool line_tracing_PE_R = false;
static bool line_tracing_PE_A = false;
static bool line_tracing_PE_stages = false;
static bool line_tracing_PE = true;
#endif

//------------------------------------------------------------
// functions used for line tracing
//------------------------------------------------------------

#ifndef __SYNTHESIS__

// to calculate total progress
template<typename T, const unsigned int ARRAY_SIZE>
T array_sum(T array[ARRAY_SIZE]) {
    T result = 0;
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        result += array[i];
    }
    return result;
}

#endif

//----------------------------------------------------
// Hardware Manipulating Helper Functions
//----------------------------------------------------

// force a register
template<class T>
T HLS_REG(T in){
#pragma HLS pipeline
#pragma HLS inline off
#pragma HLS interface port=return register
    return in;
}



bool bool_array_and(bool array1[NUM_PE_PER_HBM_CHANNEL],bool array2[NUM_PE_PER_HBM_CHANNEL]) {
    // #pragma HLS INLINE off
    bool result = true;
    for (int i = 0; i < NUM_PE_PER_HBM_CHANNEL; i++) {
        #pragma HLS UNROLL
        result = result && (array1[i] && array2[i]);
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
    return full_addr & BANK_ID_MASK;
}

// Cyclic partitioning
unsigned int get_bank_address(unsigned int full_addr) {
    return full_addr >> BANK_ID_NBITS;
}

template <typename T>
void array_shift_left(
    T array[NUM_PE_PER_HBM_CHANNEL],
    T array_swap[NUM_PE_PER_HBM_CHANNEL],
    unsigned int rotate) {
    #pragma HLS ARRAY_PARTITION variable=array_swap complete

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        array_swap[PE_idx] = array[(PE_idx + rotate) % NUM_PE_PER_HBM_CHANNEL];
    }

}

template <typename T>
void array_shift_right(T array[NUM_PE_PER_HBM_CHANNEL],unsigned int rotate) {
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

void crossbar_arbiter(
            // request address
            INDEX_T in_address[NUM_PE_PER_HBM_CHANNEL],
            bool in_valid[NUM_PE_PER_HBM_CHANNEL],
            // arbitration results
            unsigned int bank_idx_to_PE_idx[NUM_BANK_PER_HBM_CHANNEL][NUM_PORT_PER_BANK],
            unsigned int bank_address[NUM_BANK_PER_HBM_CHANNEL][NUM_PORT_PER_BANK],
            unsigned int bank_num_valid_requests[NUM_BANK_PER_HBM_CHANNEL],
            // grant PEs
            bool out_valid[NUM_PE_PER_HBM_CHANNEL],
            // priority roatae
            unsigned int rotate) {
    #pragma HLS pipeline II=1
    #pragma HLS latency min=ARBITER_LATENCY max=ARBITER_LATENCY
    assert(rotate < NUM_PE_PER_HBM_CHANNEL);

    // reset arbitration results
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

    bool in_valid_temp[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=in_valid_temp complete dim=1

    unsigned int in_address_temp[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=in_address_temp complete dim=1

    // reset arbiter results
    loop_reset_crossbar:
    for (int bank_idx = 0; bank_idx < NUM_BANK_PER_HBM_CHANNEL; bank_idx++) {
        #pragma HLS UNROLL
        for (int port_idx = 0; port_idx < NUM_PORT_PER_BANK; port_idx++) {
            #pragma HLS UNROLL
            bank_idx_to_PE_idx[bank_idx][port_idx] = 0;
            bank_address[bank_idx][port_idx] = 0;
        }
    }

    array_shift_left<bool>(in_valid, in_valid_temp, rotate);
    array_shift_left<unsigned int>(in_address, in_address_temp, rotate);

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        in_bank_idx[PE_idx] = get_bank_idx(in_address_temp[PE_idx]);
    }

    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        in_bank_address[PE_idx] = get_bank_address(in_address_temp[PE_idx]);
    }

    loop_crossbar_bank_idx:
    for (int bank_idx = 0; bank_idx < NUM_BANK_PER_HBM_CHANNEL; bank_idx++) {
        #pragma HLS UNROLL
        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            if (in_valid_temp[PE_idx] && (in_bank_idx[PE_idx] == bank_idx)) {
                // just an alias
                unsigned int port_idx = bank_num_valid_requests[bank_idx];

                // if there is idle port to assign to this PE
                if (port_idx < NUM_PORT_PER_BANK) {
                    // generate arbitration results
                    bank_idx_to_PE_idx[bank_idx][port_idx] = PE_idx;
                    bank_address[bank_idx][port_idx] = in_bank_address[PE_idx];
                    bank_num_valid_requests[bank_idx]++;
                    // grant PEs
                    out_valid[PE_idx] = true;
                }
            }
        }
    }

    array_shift_right<bool>(out_valid, rotate);
    // line tracing
    #ifndef __SYNTHESIS__
    if(line_tracing_PE_A) {
        for (int b = 0; b < NUM_BANK_PER_HBM_CHANNEL; b++) {
            for (unsigned int p = 0; p < NUM_PORT_PER_BANK; p++) {
                std::cout   << "[INFO kernel_spmv] Arbiter: Bank[" << b << "](" << std::setw(1) << bank_num_valid_requests[b] << "), "
                            << "Port[" << p <<"],"
                            << "Addr: " << std::setw(5) << ((p < bank_num_valid_requests[b]) ? std::to_string(bank_address[b][p]) : "-----") << ", "
                            << "to PE[" << std::setw(2) << ((p < bank_num_valid_requests[b]) ? std::to_string(bank_idx_to_PE_idx[b][p]) : "--") << "], "
                            << std::endl << std::flush;
            }
        }
        std::cout << std::endl;
    }
    #endif



}

void crossbar_read(
            // URAM
            VAL_T vector_uram_one_channel[VECTOR_BUFFER_LEN/NUM_BANK_PER_HBM_CHANNEL + 1][NUM_BANK_PER_HBM_CHANNEL],
            // arbitration results
            unsigned int bank_idx_to_PE_idx[NUM_BANK_PER_HBM_CHANNEL][NUM_PORT_PER_BANK],
            unsigned int bank_address[NUM_BANK_PER_HBM_CHANNEL][NUM_PORT_PER_BANK],
            unsigned int bank_num_valid_requests[NUM_BANK_PER_HBM_CHANNEL],
            // response
            VAL_T out_data[NUM_PE_PER_HBM_CHANNEL],
            // priority rotate
            unsigned int rotate) {
    #pragma HLS inline

    assert(rotate < NUM_PE_PER_HBM_CHANNEL);

    unsigned int local_bank_idx_to_PE_idx[NUM_BANK_PER_HBM_CHANNEL][NUM_PORT_PER_BANK];
    unsigned int local_bank_address[NUM_BANK_PER_HBM_CHANNEL][NUM_PORT_PER_BANK];
    unsigned int local_bank_num_valid_requests[NUM_BANK_PER_HBM_CHANNEL];
    #pragma HLS array_partition variable=local_bank_idx_to_PE_idx complete dim=0
    #pragma HLS array_partition variable=local_bank_address complete dim=0
    #pragma HLS array_partition variable=local_out_dbank_num_valid_requestsata_temp complete dim=0

    VAL_T out_data_temp[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS array_partition variable=out_data_temp complete dim=0
    // reset output
    for (int pe_idx = 0; pe_idx < NUM_PE_PER_HBM_CHANNEL; pe_idx++) {
        #pragma HLS UNROLL
        out_data_temp[pe_idx] = 0;
    }

    // input pilpeline
    for (int bank_idx = 0; bank_idx < NUM_BANK_PER_HBM_CHANNEL; bank_idx++) {
        #pragma HLS UNROLL
        local_bank_num_valid_requests[bank_idx] =
            HLS_REG<unsigned int>(
                HLS_REG<unsigned int>(
                    HLS_REG<unsigned int>(
                        HLS_REG<unsigned int>(bank_num_valid_requests[bank_idx]))));
        for (unsigned int port_idx = 0; port_idx < NUM_PORT_PER_BANK; port_idx++) {
            #pragma HLS UNROLL
            local_bank_idx_to_PE_idx[bank_idx][port_idx] =
                HLS_REG<unsigned int>(
                    HLS_REG<unsigned int>(
                        HLS_REG<unsigned int>(
                            HLS_REG<unsigned int>(bank_idx_to_PE_idx[bank_idx][port_idx]))));
            local_bank_address[bank_idx][port_idx] =
                HLS_REG<unsigned int>(
                    HLS_REG<unsigned int>(
                        HLS_REG<unsigned int>(
                            HLS_REG<unsigned int>(bank_address[bank_idx][port_idx]))));
        }
    }

    // first get read data
    for (int bank_idx = 0; bank_idx < NUM_BANK_PER_HBM_CHANNEL; bank_idx++) {
        #pragma HLS UNROLL
        for (unsigned int port_idx = 0; port_idx < NUM_PORT_PER_BANK; port_idx++) {
            #pragma HLS UNROLL
            if (port_idx < local_bank_num_valid_requests[bank_idx]) {
                out_data_temp[local_bank_idx_to_PE_idx[bank_idx][port_idx]] =
                    vector_uram_one_channel[local_bank_address[bank_idx][port_idx]][bank_idx];
            }
        }
    }

    // then send read data out
    for (int pe_idx = 0; pe_idx < NUM_PE_PER_HBM_CHANNEL; pe_idx++) {
        #pragma HLS UNROLL
        out_data[pe_idx] =
            HLS_REG<VAL_T>(
                HLS_REG<VAL_T>(
                    HLS_REG<VAL_T>(
                        HLS_REG<VAL_T>(out_data_temp[pe_idx]))));

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
        size.data[k] = x[k+1];
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
    assert(size % PACK_SIZE == 0);
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


static void compute_spmv_one_channel(hls::stream<INDEX_T> indices_stream_one_PE[NUM_PE_PER_HBM_CHANNEL], // first element is size
                                     hls::stream<VAL_T> vals_stream_one_PE[NUM_PE_PER_HBM_CHANNEL],
                                     VAL_T vector_uram_one_channel[VECTOR_BUFFER_LEN/NUM_BANK_PER_HBM_CHANNEL + 1][NUM_BANK_PER_HBM_CHANNEL],
                                     hls::stream<VAL_T> out_stream_one_channel[NUM_PE_PER_HBM_CHANNEL]) {
    // used to measure PE progress
    int size[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=size complete
    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        size[PE_idx] = indices_stream_one_PE[PE_idx].read();
    }
    int fetch_cnt[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=fetch_cnt complete
    int process_cnt[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=process_cnt complete
    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        fetch_cnt[PE_idx] = 0;
        process_cnt[PE_idx] = 0;
    }
    // PE finish flags
    bool all_fetched[NUM_PE_PER_HBM_CHANNEL];
    // This needs to be frowarded
    bool all_processed[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=all_fetched complete
    #pragma HLS ARRAY_PARTITION variable=all_processed complete
    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        all_fetched[PE_idx] = false;
        all_processed[PE_idx] = false;
    }

    // column id
    INDEX_T index[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=index complete
    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        // index[PE_idx] = indices_stream_one_PE[PE_idx].read();
        index[PE_idx] = 0;
    }

    // URAM read request valid
    bool in_valid[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=in_valid complete
    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        in_valid[PE_idx] = false;
    }

    // nnz from matrix
    VAL_T val[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=val complete

    // URAM read data (nnz from vector)
    VAL_T vector_data[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=vector_data complete

    // PE local accumulative register
    VAL_T tmp_out[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=tmp_out complete
    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        tmp_out[PE_idx] = 0;
    }

    // fetch failed flag
    bool F_valid[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=F_valid complete
    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        F_valid[PE_idx] = false;
    }

    // arbitration results
    unsigned int bank_idx_to_PE_idx[NUM_BANK_PER_HBM_CHANNEL][NUM_PORT_PER_BANK];
    unsigned int bank_address[NUM_BANK_PER_HBM_CHANNEL][NUM_PORT_PER_BANK];
    unsigned int bank_num_valid_requests[NUM_BANK_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=bank_idx_to_PE_idx complete dim=0
    #pragma HLS ARRAY_PARTITION variable=bank_address complete dim=0
    #pragma HLS ARRAY_PARTITION variable=bank_num_valid_requests complete dim=0

    // arbiter grant signal
    bool out_valid[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=out_valid complete

    // forwarding switch
    // This needs to be frowarded
    bool resend[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=resend complete
    for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
        #pragma HLS UNROLL
        resend[PE_idx] = false;
    }

    // operator of MAC
    bool write[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=write complete

    // operator of MAC
    bool accumulate[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=accumulate complete

    // forwarding channel
    // These need to be frowarded
    INDEX_T index_fwd[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=index_fwd complete
    VAL_T   val_fwd[NUM_PE_PER_HBM_CHANNEL];
    #pragma HLS ARRAY_PARTITION variable=val_fwd   complete

    // aribiter priority rotation
    unsigned int rotate = 0;
    // This needs to be frowarded
    unsigned int next_rotate = 0;

    // used for line tracing
    #ifndef __SYNTHESIS__
        int round = 0;
        bool input_success_ltr[NUM_PE_PER_HBM_CHANNEL];
        VAL_T tmp_out_ltr[NUM_PE_PER_HBM_CHANNEL];
    #endif

    loop_compute_spmv_one_channel:
    // all_processed is forwarded from P
    while (!bool_array_and(all_fetched,all_processed)) {
        #pragma HLS dependence variable=all_processed   distance=FWD_DISTANCE           inter RAW true
        #pragma HLS dependence variable=resend          distance=FWD_DISTANCE           inter RAW true
        #pragma HLS dependence variable=index_fwd       distance=FWD_DISTANCE           inter RAW true
        #pragma HLS dependence variable=val_fwd         distance=FWD_DISTANCE           inter RAW true
        #pragma HLS dependence variable=next_rotate     distance=FWD_DISTANCE           inter RAW true
        #pragma HLS PIPELINE II=1

        // line tracing
        #ifndef __SYNTHESIS__
        if(line_tracing_PE) {
            std::cout   << "[" << array_sum<int,NUM_PE_PER_HBM_CHANNEL>(process_cnt)
                        << "/" << array_sum<int,NUM_PE_PER_HBM_CHANNEL>(size)
                        << "]\t"
                        << "Round " << round
                        << std::endl;
            round ++;
        }
        #endif

        // stage F (fetch)
        rotate = next_rotate;
        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            all_fetched[PE_idx] = (fetch_cnt[PE_idx] >= size[PE_idx]);

            // no need to do anything
            if (all_fetched[PE_idx] && all_processed[PE_idx]) {
                F_valid[PE_idx] = false;
                index[PE_idx] = 0; // a dummy number
                val[PE_idx] = 0; // a dummy number
                in_valid[PE_idx] = false;
            // need to either resend or fetch or wait for the resent ones to be arbitrated
            } else {
                // resend is forwarded from P.
                if(resend[PE_idx]) {
                    F_valid[PE_idx] = true;
                    index[PE_idx] = index_fwd[PE_idx];
                    val[PE_idx] = val_fwd[PE_idx];
                    in_valid[PE_idx] = true;
                // normal fetch
                } else if(!all_fetched[PE_idx]) {
                    INDEX_T Fidx;
                    VAL_T   Fval;
                    bool Fidx_rd_success = indices_stream_one_PE[PE_idx].read_nb(Fidx);
                    bool Fval_rd_success = vals_stream_one_PE[PE_idx].read_nb(Fval);
                    if (Fidx_rd_success && Fval_rd_success) {
                        F_valid[PE_idx] = true;
                        index[PE_idx] = Fidx;
                        val[PE_idx] = Fval;
                        in_valid[PE_idx] = (Fidx != IDX_MARKER);
                        fetch_cnt[PE_idx] ++;
                    } else {
                        F_valid[PE_idx] = false;
                        index[PE_idx] = 0; // a dummy number
                        val[PE_idx] = 0; // a dummy number
                        in_valid[PE_idx] = false;
                    }

                    // used for line tracing
                    #ifndef __SYNTHESIS__
                        input_success_ltr[PE_idx] = Fidx_rd_success && Fval_rd_success;
                    #endif
                // wait for the resent ones to be arbitrated, fill in with dummies
                } else {
                    F_valid[PE_idx] = false;
                    index[PE_idx] = 0; // a dummy number
                    val[PE_idx] = 0; // a dummy number
                    in_valid[PE_idx] = false;
                }
            }


        }
        // -------- end of stage F
        // line tracing
        #ifndef __SYNTHESIS__
        if(line_tracing_PE_stages) {
            std::cout << "F" << std::flush;
        }
        #endif

        // stage A (arbiter)
        crossbar_arbiter(
            index,
            in_valid,
            bank_idx_to_PE_idx,
            bank_address,
            bank_num_valid_requests,
            out_valid,
            rotate
        );
        // -------- end of stage A
        // line tracing
        #ifndef __SYNTHESIS__
        if(line_tracing_PE_stages) {
            std::cout << "A" << std::flush;
        }
        #endif

        // stage P (post_arbiter)
        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL
            all_processed[PE_idx] = (process_cnt[PE_idx] >= size[PE_idx]);

            // activate forwarding when read will fail
            if (F_valid[PE_idx] && (!out_valid[PE_idx]) && in_valid[PE_idx]) {
                resend[PE_idx] = true;
                index_fwd[PE_idx] = index[PE_idx];
                val_fwd[PE_idx] = val[PE_idx];
            } else {
                resend[PE_idx] = false;
            }

            // select MAC operator
            if(F_valid[PE_idx]) {
                // need to write
                write[PE_idx] = !in_valid[PE_idx];
                // need to accumulate
                accumulate[PE_idx] = out_valid[PE_idx];
                // if a value will be consumed; increment process_cnt
                if(write[PE_idx] || accumulate[PE_idx]) process_cnt[PE_idx] ++;
            } else {
                write[PE_idx] = false;
                accumulate[PE_idx] = false;
            }
        }
        // update priority rotate
        next_rotate = (rotate + 1) % NUM_PE_PER_HBM_CHANNEL;
        // -------- end of stage P
        // line tracing
        #ifndef __SYNTHESIS__
        if(line_tracing_PE_stages) {
            std::cout << "P" << std::flush;
        }
        #endif

        // stage R (uram_read)
        crossbar_read(
            vector_uram_one_channel,
            bank_idx_to_PE_idx,
            bank_address,
            bank_num_valid_requests,
            vector_data,
            rotate
        );
        // -------- end of stage R
        // line tracing
        #ifndef __SYNTHESIS__
        if(line_tracing_PE_stages) {
            std::cout << "R" << std::flush;
        }
        #endif

        // stage C (commit)
        for (int PE_idx = 0; PE_idx < NUM_PE_PER_HBM_CHANNEL; PE_idx++) {
            #pragma HLS UNROLL

            // used for line tracing
            #ifndef __SYNTHESIS__
                tmp_out_ltr[PE_idx] = tmp_out[PE_idx];
            #endif

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
        // -------- end of stage C
        // line tracing
        #ifndef __SYNTHESIS__
        if(line_tracing_PE_stages) {
            std::cout << "C" << std::endl << std::flush;
        }
        #endif


        // line tracing
        #ifndef __SYNTHESIS__
        if(line_tracing_PE) {
            // PE states
            for (size_t k = 0; k < NUM_PE_PER_HBM_CHANNEL; k++) {
                bool display_value = input_success_ltr[k] || !out_valid[k];
                bool is_marker = (index[k] == IDX_MARKER);
                std::cout << "PE [" << std::setw(2) << k << "] {"
                                    << ""  << std::setw(4) << (all_processed[k]     ?  "----" :  std::to_string(process_cnt[k]))                                       << ""  << ""
                                    << "} {"
                                    << ""  << std::setw(2) << (resend[k]            ? "R>" : (input_success_ltr[k] ? "->" : "--"))                              << ""  << "|"
                                    << ""  << std::setw(9) << (display_value        ? std::to_string((float)val[k])                                     : "--") << ""  << "|"
                                    << ""  << std::setw(9) << (display_value        ? std::to_string((float)vector_data[k])                             : "--") << ""  << "|"
                                    << "[" << std::setw(5) << (display_value        ? (!is_marker ? std::to_string(index[k])                : "-EOR-")   : "--") << ""  << ""
                                    << "(" << std::setw(2) << (display_value        ? (!is_marker ? std::to_string(index[k] & BANK_ID_MASK) : "..")      : "--") << ")]"<< "|"
                                    << ""  << std::setw(9) << (!all_processed[k]    ? std::to_string((float)tmp_out_ltr[k])                             : "--") << ""  << "|"
                                    << ""  << std::setw(9) << (!all_processed[k]    ? std::to_string((float)tmp_out[k])                                 : "--") << ""
                                    << "} {"
                                    << (index[k] == IDX_MARKER  ? "Rm" :
                                            in_valid[k]         ? "Rv" : " ." ) << ":"
                                    << (!out_valid[k]           ?  "x" :  "o" ) << "|"
                                    << (write[k]                ? "Wc" :
                                            accumulate[k]       ? "Ac" : " ." ) << "|"
                                    << "}";
                std::cout << std::endl << std::flush;
            }
        }
        #endif
    }
}

// might need pipelining here
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


// #define WRITE_OUT_BRAM_ONE_CHANNEL(streams, brams, channel_idx, num_rows) { \
//     unsigned int start_PE_idx = channel_idx * NUM_PE_PER_HBM_CHANNEL; \
//     write_out_bram_one_PE(streams[0], brams[start_PE_idx + 0], start_PE_idx + 0, num_rows); \
//     write_out_bram_one_PE(streams[1], brams[start_PE_idx + 1], start_PE_idx + 1, num_rows); \
//     write_out_bram_one_PE(streams[2], brams[start_PE_idx + 2], start_PE_idx + 2, num_rows); \
//     write_out_bram_one_PE(streams[3], brams[start_PE_idx + 3], start_PE_idx + 3, num_rows); \
//     write_out_bram_one_PE(streams[4], brams[start_PE_idx + 4], start_PE_idx + 4, num_rows); \
//     write_out_bram_one_PE(streams[5], brams[start_PE_idx + 5], start_PE_idx + 5, num_rows); \
//     write_out_bram_one_PE(streams[6], brams[start_PE_idx + 6], start_PE_idx + 6, num_rows); \
//     write_out_bram_one_PE(streams[7], brams[start_PE_idx + 7], start_PE_idx + 7, num_rows); \
// } \


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
    // const INDEX_T *channel_8_partition_indptr,
    // const PACKET_T *channel_8_matrix,
    // const INDEX_T *channel_9_partition_indptr,
    // const PACKET_T *channel_9_matrix,
    // const INDEX_T *channel_10_partition_indptr,
    // const PACKET_T *channel_10_matrix,
    // const INDEX_T *channel_11_partition_indptr,
    // const PACKET_T *channel_11_matrix,
    // const INDEX_T *channel_12_partition_indptr,
    // const PACKET_T *channel_12_matrix,
    // const INDEX_T *channel_13_partition_indptr,
    // const PACKET_T *channel_13_matrix,
    // const INDEX_T *channel_14_partition_indptr,
    // const PACKET_T *channel_14_matrix,
    // const INDEX_T *channel_15_partition_indptr,
    // const PACKET_T *channel_15_matrix,
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
// #pragma HLS INTERFACE m_axi port=channel_8_matrix offset=slave bundle=gmem8
// #pragma HLS INTERFACE m_axi port=channel_9_matrix offset=slave bundle=gmem9
// #pragma HLS INTERFACE m_axi port=channel_10_matrix offset=slave bundle=gmem10
// #pragma HLS INTERFACE m_axi port=channel_11_matrix offset=slave bundle=gmem11
// #pragma HLS INTERFACE m_axi port=channel_12_matrix offset=slave bundle=gmem12
// #pragma HLS INTERFACE m_axi port=channel_13_matrix offset=slave bundle=gmem13
// #pragma HLS INTERFACE m_axi port=channel_14_matrix offset=slave bundle=gmem14
// #pragma HLS INTERFACE m_axi port=channel_15_matrix offset=slave bundle=gmem15

#pragma HLS INTERFACE m_axi port=channel_0_partition_indptr offset=slave bundle=gmem16
#pragma HLS INTERFACE m_axi port=channel_1_partition_indptr offset=slave bundle=gmem16
#pragma HLS INTERFACE m_axi port=channel_2_partition_indptr offset=slave bundle=gmem16
#pragma HLS INTERFACE m_axi port=channel_3_partition_indptr offset=slave bundle=gmem16
#pragma HLS INTERFACE m_axi port=channel_4_partition_indptr offset=slave bundle=gmem16
#pragma HLS INTERFACE m_axi port=channel_5_partition_indptr offset=slave bundle=gmem16
#pragma HLS INTERFACE m_axi port=channel_6_partition_indptr offset=slave bundle=gmem16
#pragma HLS INTERFACE m_axi port=channel_7_partition_indptr offset=slave bundle=gmem16
// #pragma HLS INTERFACE m_axi port=channel_8_partition_indptr offset=slave bundle=gmem16
// #pragma HLS INTERFACE m_axi port=channel_9_partition_indptr offset=slave bundle=gmem16
// #pragma HLS INTERFACE m_axi port=channel_10_partition_indptr offset=slave bundle=gmem16
// #pragma HLS INTERFACE m_axi port=channel_11_partition_indptr offset=slave bundle=gmem16
// #pragma HLS INTERFACE m_axi port=channel_12_partition_indptr offset=slave bundle=gmem16
// #pragma HLS INTERFACE m_axi port=channel_13_partition_indptr offset=slave bundle=gmem16
// #pragma HLS INTERFACE m_axi port=channel_14_partition_indptr offset=slave bundle=gmem16
// #pragma HLS INTERFACE m_axi port=channel_15_partition_indptr offset=slave bundle=gmem16

#pragma HLS INTERFACE m_axi port=vector offset=slave bundle=gmem17
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem17
#if defined(USE_MASK)
#pragma HLS INTERFACE m_axi port=mask offset=slave bundle=gmem17
#endif

#pragma HLS INTERFACE s_axilite port=channel_0_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_1_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_2_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_3_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_4_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_5_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_6_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=channel_7_matrix bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_8_matrix bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_9_matrix bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_10_matrix bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_11_matrix bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_12_matrix bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_13_matrix bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_14_matrix bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_15_matrix bundle=control

#pragma HLS INTERFACE s_axilite port=channel_0_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_1_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_2_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_3_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_4_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_5_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_6_partition_indptr bundle=control
#pragma HLS INTERFACE s_axilite port=channel_7_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_8_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_9_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_10_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_11_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_12_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_13_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_14_partition_indptr bundle=control
// #pragma HLS INTERFACE s_axilite port=channel_15_partition_indptr bundle=control

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
// #pragma HLS DATA_PACK variable=channel_8_matrix
// #pragma HLS DATA_PACK variable=channel_9_matrix
// #pragma HLS DATA_PACK variable=channel_10_matrix
// #pragma HLS DATA_PACK variable=channel_11_matrix
// #pragma HLS DATA_PACK variable=channel_12_matrix
// #pragma HLS DATA_PACK variable=channel_13_matrix
// #pragma HLS DATA_PACK variable=channel_14_matrix
// #pragma HLS DATA_PACK variable=channel_15_matrix

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

    INDEX_T partition_indptr_bram[NUM_HBM_CHANNEL][MAX_NUM_PARTITION * (PACK_SIZE + 1)];
    #pragma HLS ARRAY_PARTITION variable=partition_indptr_bram complete dim=1
    assert(num_row_partitions * num_col_partitions <= MAX_NUM_PARTITION);

    int size_partition_indptr = num_row_partitions * num_col_partitions * (PACK_SIZE + 1);

    for (int i = 0; i < size_partition_indptr; i++) {
        partition_indptr_bram[0][i] = channel_0_partition_indptr[i];
    }
    for (int i = 0; i < size_partition_indptr; i++) {
        partition_indptr_bram[1][i] = channel_1_partition_indptr[i];
    }
    for (int i = 0; i < size_partition_indptr; i++) {
        partition_indptr_bram[2][i] = channel_2_partition_indptr[i];
    }
    for (int i = 0; i < size_partition_indptr; i++) {
        partition_indptr_bram[3][i] = channel_3_partition_indptr[i];
    }
    for (int i = 0; i < size_partition_indptr; i++) {
        partition_indptr_bram[4][i] = channel_4_partition_indptr[i];
    }
    for (int i = 0; i < size_partition_indptr; i++) {
        partition_indptr_bram[5][i] = channel_5_partition_indptr[i];
    }
    for (int i = 0; i < size_partition_indptr; i++) {
        partition_indptr_bram[6][i] = channel_6_partition_indptr[i];
    }
    for (int i = 0; i < size_partition_indptr; i++) {
        partition_indptr_bram[7][i] = channel_7_partition_indptr[i];
    }
    // for (int i = 0; i < size_partition_indptr; i++) {
    //     partition_indptr_bram[8][i] = channel_8_partition_indptr[i];
    // }
    // for (int i = 0; i < size_partition_indptr; i++) {
    //     partition_indptr_bram[9][i] = channel_9_partition_indptr[i];
    // }
    // for (int i = 0; i < size_partition_indptr; i++) {
    //     partition_indptr_bram[10][i] = channel_10_partition_indptr[i];
    // }
    // for (int i = 0; i < size_partition_indptr; i++) {
    //     partition_indptr_bram[11][i] = channel_11_partition_indptr[i];
    // }
    // for (int i = 0; i < size_partition_indptr; i++) {
    //     partition_indptr_bram[12][i] = channel_12_partition_indptr[i];
    // }
    // for (int i = 0; i < size_partition_indptr; i++) {
    //     partition_indptr_bram[13][i] = channel_13_partition_indptr[i];
    // }
    // for (int i = 0; i < size_partition_indptr; i++) {
    //     partition_indptr_bram[14][i] = channel_14_partition_indptr[i];
    // }
    // for (int i = 0; i < size_partition_indptr; i++) {
    //     partition_indptr_bram[15][i] = channel_15_partition_indptr[i];
    // }

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
                                    &partition_indptr_bram[0][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_1_matrix,
                                    indices_stream[1],
                                    vals_stream[1],
                                    &partition_indptr_bram[1][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_2_matrix,
                                    indices_stream[2],
                                    vals_stream[2],
                                    &partition_indptr_bram[2][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_3_matrix,
                                    indices_stream[3],
                                    vals_stream[3],
                                    &partition_indptr_bram[3][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_4_matrix,
                                    indices_stream[4],
                                    vals_stream[4],
                                    &partition_indptr_bram[4][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_5_matrix,
                                    indices_stream[5],
                                    vals_stream[5],
                                    &partition_indptr_bram[5][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_6_matrix,
                                    indices_stream[6],
                                    vals_stream[6],
                                    &partition_indptr_bram[6][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            read_matrix_one_channel(channel_7_matrix,
                                    indices_stream[7],
                                    vals_stream[7],
                                    &partition_indptr_bram[7][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            // read_matrix_one_channel(channel_8_matrix,
            //                         indices_stream[8],
            //                         vals_stream[8],
            //                         &partition_indptr_bram[8][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            // read_matrix_one_channel(channel_9_matrix,
            //                         indices_stream[9],
            //                         vals_stream[9],
            //                         &partition_indptr_bram[9][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            // read_matrix_one_channel(channel_10_matrix,
            //                         indices_stream[10],
            //                         vals_stream[10],
            //                         &partition_indptr_bram[10][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            // read_matrix_one_channel(channel_11_matrix,
            //                         indices_stream[11],
            //                         vals_stream[11],
            //                         &partition_indptr_bram[11][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            // read_matrix_one_channel(channel_12_matrix,
            //                         indices_stream[12],
            //                         vals_stream[12],
            //                         &partition_indptr_bram[12][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            // read_matrix_one_channel(channel_13_matrix,
            //                         indices_stream[13],
            //                         vals_stream[13],
            //                         &partition_indptr_bram[13][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            // read_matrix_one_channel(channel_14_matrix,
            //                         indices_stream[14],
            //                         vals_stream[14],
            //                         &partition_indptr_bram[14][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);
            // read_matrix_one_channel(channel_15_matrix,
            //                         indices_stream[15],
            //                         vals_stream[15],
            //                         &partition_indptr_bram[15][(row_partition_idx*num_col_partitions + col_partition_idx)*(PACK_SIZE+1)]);

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
            // unpack_matrix_one_channel(indices_stream[8],
            //                           unpacked_indices_stream[8],
            //                           vals_stream[8],
            //                           unpacked_vals_stream[8]);
            // unpack_matrix_one_channel(indices_stream[9],
            //                           unpacked_indices_stream[9],
            //                           vals_stream[9],
            //                           unpacked_vals_stream[9]);
            // unpack_matrix_one_channel(indices_stream[10],
            //                           unpacked_indices_stream[10],
            //                           vals_stream[10],
            //                           unpacked_vals_stream[10]);
            // unpack_matrix_one_channel(indices_stream[11],
            //                           unpacked_indices_stream[11],
            //                           vals_stream[11],
            //                           unpacked_vals_stream[11]);
            // unpack_matrix_one_channel(indices_stream[12],
            //                           unpacked_indices_stream[12],
            //                           vals_stream[12],
            //                           unpacked_vals_stream[12]);
            // unpack_matrix_one_channel(indices_stream[13],
            //                           unpacked_indices_stream[13],
            //                           vals_stream[13],
            //                           unpacked_vals_stream[13]);
            // unpack_matrix_one_channel(indices_stream[14],
            //                           unpacked_indices_stream[14],
            //                           vals_stream[14],
            //                           unpacked_vals_stream[14]);
            // unpack_matrix_one_channel(indices_stream[15],
            //                           unpacked_indices_stream[15],
            //                           vals_stream[15],
            //                           unpacked_vals_stream[15]);

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
            // compute_spmv_one_channel(unpacked_indices_stream[8],
            //                          unpacked_vals_stream[8],
            //                          vector_uram[8],
            //                          out_stream[8]);
            // compute_spmv_one_channel(unpacked_indices_stream[9],
            //                          unpacked_vals_stream[9],
            //                          vector_uram[9],
            //                          out_stream[9]);
            // compute_spmv_one_channel(unpacked_indices_stream[10],
            //                          unpacked_vals_stream[10],
            //                          vector_uram[10],
            //                          out_stream[10]);
            // compute_spmv_one_channel(unpacked_indices_stream[11],
            //                          unpacked_vals_stream[11],
            //                          vector_uram[11],
            //                          out_stream[11]);
            // compute_spmv_one_channel(unpacked_indices_stream[12],
            //                          unpacked_vals_stream[12],
            //                          vector_uram[12],
            //                          out_stream[12]);
            // compute_spmv_one_channel(unpacked_indices_stream[13],
            //                          unpacked_vals_stream[13],
            //                          vector_uram[13],
            //                          out_stream[13]);
            // compute_spmv_one_channel(unpacked_indices_stream[14],
            //                          unpacked_vals_stream[14],
            //                          vector_uram[14],
            //                          out_stream[14]);
            // compute_spmv_one_channel(unpacked_indices_stream[15],
            //                          unpacked_vals_stream[15],
            //                          vector_uram[15],
            //                          out_stream[15]);

            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[0], out_bram, 0, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[1], out_bram, 1, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[2], out_bram, 2, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[3], out_bram, 3, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[4], out_bram, 4, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[5], out_bram, 5, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[6], out_bram, 6, size)
            WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[7], out_bram, 7, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[8], out_bram, 8, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[9], out_bram, 9, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[10], out_bram, 10, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[11], out_bram, 11, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[12], out_bram, 12, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[13], out_bram, 13, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[14], out_bram, 14, size)
            // WRITE_OUT_BRAM_ONE_CHANNEL(out_stream[15], out_bram, 15, size)
        }

        assert(size % PACK_SIZE == 0);
        unsigned int vsize = size / PACK_SIZE;
        PACKED_VAL_T tmp_out;
#if defined(USE_MASK)
        PACKED_VAL_T tmp_mask;
#endif

        // might need pipelining here
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
