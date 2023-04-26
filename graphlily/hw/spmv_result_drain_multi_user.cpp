#include <hls_stream.h>
#include <ap_int.h>

#include "libfpga/hisparse.h"

#ifndef __SYNTHESIS__
#include <iostream>
#include <cstdlib>
// #define SPMV_RESULT_DRAIN_LINE_TRACING
#endif

extern "C" {
void spmv_result_drain_user(
    PACKED_VAL_T *packed_dense_result,      // out
    const PACKED_VAL_T *packed_dense_mask,  // in
    const unsigned row_part_id,             // in
    // const unsigned rows_per_c_in_partition, // in
    const unsigned zero_ufixed,             // in
    const MASK_T mask_type,                 // in

    // TODO: When adding scheduling, take this away because it will no longer pull from all three
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

    //* User A:
    char current_input_uA = 0;
    ap_uint<3> finished_uA = 0;
    char counter_uA = 0;
    unsigned write_counter_uA = 0;
    bool exit_uA = false;
    // TODO: update row_part_id (and PACK_SIZE?) for each user
    unsigned pkt_idx_offset_uA = row_part_id * LOGICAL_OB_SIZE / PACK_SIZE;
    
    //* User B:
    char current_input_uB = 0;
    ap_uint<3> finished_uB = 0;
    char counter_uB = 0;
    unsigned write_counter_uB = 0;
    bool exit_uB = false;
    // TODO: update row_part_id (and PACK_SIZE?) for each user
    unsigned pkt_idx_offset_uB = row_part_id * LOGICAL_OB_SIZE / PACK_SIZE;
    
    //* Exit for when A and B finish result drain
    bool exit = true;
    
    
    result_drain_main_loop:
    while (!exit) {
        #pragma HLS pipeline II=1
        VEC_AXIS_T pkt;
        bool do_write_uA = false;
        bool do_write_uB = false;

        //* ================================================================
        //* USER A results
        //* ================================================================
        switch (current_input_uA) {
            case 0:
                if (counter_uA < SK0_CLUSTER && !finished_uA[0]) {
                    pkt = from_SLR0.read();
#ifdef SPMV_RESULT_DRAIN_LINE_TRACING
                    std::cout << "RD got pkt from SK0: " << pkt;
#endif
                    if (pkt.user[0] == EOS) {
                        finished_uA[0] = true;
                        do_write_uA = false;
                    } else if (pkt.user != SOD && pkt.user != EOD) {
                        do_write_uA = true;
                    }
                    counter_uA++;
                } else {
                    do_write_uA = false;
                    current_input_uA = 1;
                    counter_uA = 0;
                }
                break;
            case 1:
                if (counter_uA < SK1_CLUSTER && !finished_uA[1]) {
                    pkt = from_SLR1.read();
#ifdef SPMV_RESULT_DRAIN_LINE_TRACING
                    std::cout << "RD got pkt from SK1: " << pkt;
#endif
                    if (pkt.user == EOS) {
                        finished_uA[1] = true;
                        do_write_uA = false;
                    } else if (pkt.user != SOD && pkt.user != EOD) {
                        do_write_uA = true;
                    }
                    counter_uA++;
                } else {
                    do_write_uA = false;
                    current_input_uA = 2;
                    counter_uA = 0;
                }
                break;
            case 2:
                if (counter < SK2_CLUSTER && !finished_uA[2]) {
                    pkt = from_SLR2.read();
#ifdef SPMV_RESULT_DRAIN_LINE_TRACING
                    std::cout << "RD got pkt from SK2: " << pkt;
#endif
                    if (pkt.user == EOS) {
                        finished_uA[2] = true;
                        do_write_uA = false;
                    } else if (pkt.user != SOD && pkt.user != EOD) {
                        do_write_uA = true;
                    }
                    counter_uA++;
                } else {
                    do_write_uA = false;
                    current_input_uA = 0;
                    counter_uA = 0;
                }
                break;
            default: break;
        } 

        // switch (current_input)
        exit_uA = finished_uA.and_reduce();

        unsigned abs_pkt_idx_uA = write_counter_uA + pkt_idx_offset_uA;
        if (do_write_uA) {
            PACKED_VAL_T rout;
            PACKED_VAL_T mask;
            if (mask_type != NOMASK) {
                mask = packed_dense_mask[abs_pkt_idx_uA];
            }
            for (unsigned k = 0; k < PACK_SIZE; k++) {
                #pragma HLS unroll
                bool do_write_m_uA = false; // do write after checking mask
                switch (mask_type) {
                    case NOMASK:
                        do_write_m_uA = true;
                        break;
                    case WRITETOONE:
                        do_write_m_uA = (mask.data[k] != 0);
                        break;
                    case WRITETOZERO:
                        do_write_m_uA = (mask.data[k] == 0);
                        break;
                    default:
                        do_write_m_uA = false;
                        #ifndef __SYNTHESIS__
                        std::cout << "Invalid mask type" << std::endl;
                        std::exit(EXIT_FAILURE);
                        #endif
                        break;
                }
                if (do_write_m_uA) {
                    VAL_T_BITCAST(rout.data[k]) = VEC_AXIS_VAL(pkt, k);
                } else {
                    VAL_T_BITCAST(rout.data[k]) = VAL_T_BITCAST(zero);
                }
            }
            write_counter_uA++;
            packed_dense_result[abs_pkt_idx_uA] = rout;
        }

#ifdef SPMV_RESULT_DRAIN_LINE_TRACING
        if (do_write_uA) {
            std::cout << ", written to " << abs_pkt_idx_uA << std::endl;
        } else {
            std::cout << std::endl;
        }
#endif

        //* ================================================================
        //* USER B results
        //* ================================================================
        switch (current_input_uB) {
            case 0:
                if (counter_uB < SK0_CLUSTER && !finished_uB[0]) {
                    pkt = from_SLR0.read();
#ifdef SPMV_RESULT_DRAIN_LINE_TRACING
                    std::cout << "RD got pkt from SK0: " << pkt;
#endif
                    if (pkt.user[0] == EOS) {
                        finished_uB[0] = true;
                        do_write_uB = false;
                    } else if (pkt.user != SOD && pkt.user != EOD) {
                        do_write_uB = true;
                    }
                    counter_uB++;
                } else {
                    do_write_uB = false;
                    current_input_uB = 1;
                    counter_uB = 0;
                }
                break;
            case 1:
                if (counter_uB < SK1_CLUSTER && !finished_uB[1]) {
                    pkt = from_SLR1.read();
#ifdef SPMV_RESULT_DRAIN_LINE_TRACING
                    std::cout << "RD got pkt from SK1: " << pkt;
#endif
                    if (pkt.user == EOS) {
                        finished_uB[1] = true;
                        do_write_uB = false;
                    } else if (pkt.user != SOD && pkt.user != EOD) {
                        do_write_uB = true;
                    }
                    counter_uB++;
                } else {
                    do_write_uB = false;
                    current_input_uB = 2;
                    counter_uB = 0;
                }
                break;
            case 2:
                if (counter_uB < SK2_CLUSTER && !finished_uB[2]) {
                    pkt = from_SLR2.read();
#ifdef SPMV_RESULT_DRAIN_LINE_TRACING
                    std::cout << "RD got pkt from SK2: " << pkt;
#endif
                    if (pkt.user == EOS) {
                        finished_uB[2] = true;
                        do_write_uB = false;
                    } else if (pkt.user != SOD && pkt.user != EOD) {
                        do_write_uB = true;
                    }
                    counter_uB++;
                } else {
                    do_write_uB = false;
                    current_input_uB = 0;
                    counter_uB = 0;
                }
                break;
            default: break;
        }
        
        // switch (current_input)
        exit_ub = finished_uB.and_reduce();

        unsigned abs_pkt_idx_uB = write_counter_uB + pkt_idx_offset_uB;
        if (do_write_uB) {
            PACKED_VAL_T rout;
            PACKED_VAL_T mask;
            if (mask_type != NOMASK) {
                mask = packed_dense_mask[abs_pkt_idx_uB];
            }
            for (unsigned k = 0; k < PACK_SIZE; k++) {
                #pragma HLS unroll
                bool do_write_m_uB = false; // do write after checking mask
                switch (mask_type) {
                    case NOMASK:
                        do_write_m_uB = true;
                        break;
                    case WRITETOONE:
                        do_write_m_uB = (mask.data[k] != 0);
                        break;
                    case WRITETOZERO:
                        do_write_m_uB = (mask.data[k] == 0);
                        break;
                    default:
                        do_write_m_uB = false;
                        #ifndef __SYNTHESIS__
                        std::cout << "Invalid mask type" << std::endl;
                        std::exit(EXIT_FAILURE);
                        #endif
                        break;
                }
                if (do_write_m_uB) {
                    VAL_T_BITCAST(rout.data[k]) = VEC_AXIS_VAL(pkt, k);
                } else {
                    VAL_T_BITCAST(rout.data[k]) = VAL_T_BITCAST(zero);
                }
            }
            write_counter_uB++;
            packed_dense_result[abs_pkt_idx_uB] = rout;
        }

#ifdef SPMV_RESULT_DRAIN_LINE_TRACING
        if (do_write_uB) {
            std::cout << ", written to " << abs_pkt_idx_uB << std::endl;
        } else {
            std::cout << std::endl;
        }
#endif


        //! Exit only if both A and B are done
        if ((exit_uA == true) && (exit_uB == true)) {
            exit = true;
        }

    }

} // kernel
} // extern "C"