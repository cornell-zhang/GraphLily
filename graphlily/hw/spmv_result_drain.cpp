#include <hls_stream.h>
#include <ap_int.h>

#include "libfpga/hisparse.h"

#ifndef __SYNTHESIS__
#include <iostream>
#include <cstdlib>
// #define SPMV_RESULT_DRAIN_LINE_TRACING
#endif

extern "C" {
void spmv_result_drain(
    PACKED_VAL_T *packed_dense_result,      // out
    PACKED_VAL_T *packed_dense_mask,        // in
    const unsigned row_part_id,             // in
    // const unsigned rows_per_c_in_partition, // in
    unsigned zero_ufixed,                   // in
    MASK_T mask_type,                       // in
    hls::stream<VEC_AXIS_T> &from_SLR0,     // out
    hls::stream<VEC_AXIS_T> &from_SLR1,     // out
    hls::stream<VEC_AXIS_T> &from_SLR2      // out
) {
    #pragma HLS interface m_axi port=packed_dense_result offset=slave bundle=spmv_vin
    #pragma HLS interface m_axi port=packed_dense_mask offset=slave bundle=spmv_mask
    #pragma HLS interface s_axilite port=packed_dense_result bundle=control
    #pragma HLS interface s_axilite port=packed_dense_mask bundle=control
    #pragma HLS interface s_axilite port=row_part_id bundle=control
    #pragma HLS interface s_axilite port=zero_ufixed bundle=control
    #pragma HLS interface s_axilite port=mask_type bundle=control
    #pragma HLS interface s_axilite port=return bundle=control

    #pragma HLS interface axis register both port=from_SLR0
    #pragma HLS interface axis register both port=from_SLR1
    #pragma HLS interface axis register both port=from_SLR2

    // TODO: maunally handle burst write?

    // zero value for mask
    VAL_T zero;
    LOAD_RAW_BITS_FROM_UINT(zero, zero_ufixed);

    // write back
    char current_input = 0;
    ap_uint<3> finished = 0;
    char counter = 0;
    unsigned write_counter = 0;
    bool exit = false;
    unsigned pkt_idx_offset = row_part_id * LOGICAL_OB_SIZE / PACK_SIZE;
    result_drain_main_loop:
    while (!exit) {
        #pragma HLS pipeline II=1
        VEC_AXIS_T pkt;
        bool do_write = false;
        switch (current_input) {
            case 0:
                if (counter < SK0_CLUSTER && !finished[0]) {
                    pkt = from_SLR0.read();
#ifdef SPMV_RESULT_DRAIN_LINE_TRACING
                    std::cout << "RD got pkt from SK0: " << pkt;
#endif
                    if (pkt.user == EOS) {
                        finished[0] = true;
                        do_write = false;
                    } else if (pkt.user != SOD && pkt.user != EOD) {
                        do_write = true;
                    }
                    counter++;
                } else {
                    do_write = false;
                    current_input = 1;
                    counter = 0;
                }
                break;
            case 1:
                if (counter < SK1_CLUSTER && !finished[1]) {
                    pkt = from_SLR1.read();
#ifdef SPMV_RESULT_DRAIN_LINE_TRACING
                    std::cout << "RD got pkt from SK1: " << pkt;
#endif
                    if (pkt.user == EOS) {
                        finished[1] = true;
                        do_write = false;
                    } else if (pkt.user != SOD && pkt.user != EOD) {
                        do_write = true;
                    }
                    counter++;
                } else {
                    do_write = false;
                    current_input = 2;
                    counter = 0;
                }
                break;
            case 2:
                if (counter < SK2_CLUSTER && !finished[2]) {
                    pkt = from_SLR2.read();
#ifdef SPMV_RESULT_DRAIN_LINE_TRACING
                    std::cout << "RD got pkt from SK2: " << pkt;
#endif
                    if (pkt.user == EOS) {
                        finished[2] = true;
                        do_write = false;
                    } else if (pkt.user != SOD && pkt.user != EOD) {
                        do_write = true;
                    }
                    counter++;
                } else {
                    do_write = false;
                    current_input = 0;
                    counter = 0;
                }
                break;
            default: break;
        } // switch (current_input)
        exit = finished.and_reduce();

        unsigned abs_pkt_idx = write_counter + pkt_idx_offset;
        if (do_write) {
            PACKED_VAL_T rout;
            PACKED_VAL_T mask;
            if (mask_type != NOMASK) {
                mask = packed_dense_mask[abs_pkt_idx];
            }
            for (unsigned k = 0; k < PACK_SIZE; k++) {
                #pragma HLS unroll
                bool do_write_m = false; // do write after checking mask
                switch (mask_type) {
                    case NOMASK:
                        do_write_m = true;
                        break;
                    case WRITETOONE:
                        do_write_m = (mask.data[k] != zero);
                        break;
                    case WRITETOZERO:
                        do_write_m = (mask.data[k] == zero);
                        break;
                    default:
                        do_write_m = false;
                        #ifndef __SYNTHESIS__
                        std::cout << "Invalid mask type" << std::endl;
                        std::exit(EXIT_FAILURE);
                        #endif
                        break;
                }
                if (do_write_m) {
                    VAL_T_BITCAST(rout.data[k]) = VEC_AXIS_VAL(pkt, k);
                } else {
                    VAL_T_BITCAST(rout.data[k]) = VAL_T_BITCAST(zero);
                }
            }
            write_counter++;
            packed_dense_result[abs_pkt_idx] = rout;
        }

#ifdef SPMV_RESULT_DRAIN_LINE_TRACING
        if (do_write) {
            std::cout << ", written to " << abs_pkt_idx << std::endl;
        } else {
            std::cout << std::endl;
        }
#endif

    }

} // kernel
} // extern "C"
