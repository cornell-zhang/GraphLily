#ifndef GRAPHLILY_HW_UFIXED_PE_H_
#define GRAPHLILY_HW_UFIXED_PE_H_

#include "hls_stream.h"
#include <iostream>
#include <iomanip>

#include "libfpga/hisparse.h"
#include "libfpga/math_constants.h"
#include "libfpga/util.h"

#ifndef __SYNTHESIS__
bool line_tracing_ufixed_pe_cluster = true;
bool sw_emu_early_abort = false;
unsigned sw_emu_iter_limit = 100;
#endif

#define MIN(a, b) ((a < b)? a : b)

//----------------------------------------------------------------
// ALUs
//----------------------------------------------------------------

template<typename ValT, typename OpT>
ValT pe_ufixed_mul_alu(ValT a, ValT b, ValT z, OpT op, bool en) {
    #pragma HLS pipeline II=1
    #pragma HLS latency min=2 max=2
    ValT out;
    switch (op) {
        case MULADD:
            out = a * b;
            break;
        case ANDOR:
            out = a && b;
            break;
        case ADDMIN:
            out = a + b;
            break;
        default:
            out = z;  // z is the zero value in this semiring
            break;
    }
    return en ? out : z;
}

template<typename ValT, typename OpT>
ValT pe_ufixed_add_alu(ValT a, ValT b, ValT z, OpT op, bool en) {
    #pragma HLS pipeline II=1
    #pragma HLS latency min=0 max=0
    ValT out;
    switch (op) {
        case MULADD:
            out = a + b;
            break;
        case ANDOR:
            out = a || b;
            break;
        case ADDMIN:
            out = MIN(a, b);
            break;
        default:
            out = z;
            break;
    }
    return en ? out : z;
}

//----------------------------------------------------------------
// unsigned fixed-point pe_cluster with forwarding logic
//----------------------------------------------------------------

/*
  The fixed-point PE is divided into 2 parts. These two parts are connected via a FIFO.
  part 1: Fetch the payload from previous stage, and do the <x> operator in the semiring.
          This part has no data dependencies.
  part 2: Read the current value from the output buffer, do the <+> operator and wirte back
          to the output buffer. This part has data dependencies.
  If we merge these two parts, we still need a skid buffer since it is not possible to stall the
  fixed-point multiplier manually in HLS. We have to use a FIFO to apply back pressure
  and the tool will implement the stall logic.
  With full-forwarding, we do not need to stall anymore, but having this 2-part architecture allow as to add
  stall logic if necessary.
*/
template<typename ValT>
struct PP_STREAM_T {
    unsigned addr;
    ValT incr;
};

// ufixed pe cluster part 1
template<typename ValT, typename OpT, typename PayloadT, unsigned num_PE>
void ufixed_pe_cluster_part1(
    hls::stream<PayloadT> input_payloads[num_PE],
    hls::stream<PP_STREAM_T<ValT> > output_payloads[num_PE],
    hls::stream<unsigned> &num_payloads_in,
    hls::stream<unsigned> &num_payloads_out,
    OpT Op,
    ValT Zero
) {
    // loop control
    bool prev_finish = false;
    unsigned num_payload = 0;
    unsigned process_cnt = 0;
    bool loop_exit = false;

    #ifndef __SYNTHESIS__
    int sw_emu_iter_cnt_p1 = 0;
    #endif

    // pipeline registers
    // F stage (depth 1)
    ValT a_F[num_PE];
    #pragma HLS array_partition variable=a_F complete
    ValT b_F[num_PE];
    #pragma HLS array_partition variable=b_F complete
    unsigned addr_F[num_PE];
    #pragma HLS array_partition variable=addr_F complete
    bool valid_F[num_PE];
    #pragma HLS array_partition variable=valid_F complete
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        a_F[i] = 0;
        b_F[i] = 0;
        addr_F[i] = 0;
        valid_F[i] = false;
    }

    // M stage (depth 8)
    ValT incr_M[num_PE];
    #pragma HLS array_partition variable=incr_M complete
    unsigned addr_M[num_PE];
    #pragma HLS array_partition variable=addr_M complete
    bool valid_M[num_PE];
    #pragma HLS array_partition variable=valid_M complete
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        incr_M[i] = 0;
        addr_M[i] = 0;
        valid_M[i] = false;
    }

    loop_pe_part1:
    while (!loop_exit) {
        #pragma HLS pipeline II=1
        #pragma HLS latency min=4 max=4
        #pragma HLS dependence variable=loop_exit inter RAW true distance=5

        // Fetch stage (F)
        loop_F:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            PayloadT payload_in_tmp;
            bool fifo_empty = !(input_payloads[PEid].read_nb(payload_in_tmp));
            if (fifo_empty) {
                a_F[PEid] = 0;
                b_F[PEid] = 0;
                addr_F[PEid] = 0;
                valid_F[PEid] = false;
            } else {
                a_F[PEid] = payload_in_tmp.data.mat_val;
                b_F[PEid] = payload_in_tmp.data.vec_val;
                addr_F[PEid] = payload_in_tmp.index;
                valid_F[PEid] = true;
            }
        }
        // ----- end of F stage

        // Mul stage (M)
        unsigned process_cnt_incr = 0;
        loop_M:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            valid_M[PEid] = valid_F[PEid];
            addr_M[PEid] = addr_F[PEid];
            incr_M[PEid] = pe_ufixed_mul_alu<ValT, OpT>(a_F[PEid], b_F[PEid], Zero, Op, valid_M[PEid]);
            if (valid_M[PEid]) {
                PP_STREAM_T<ValT> payload_out_tmp;
                payload_out_tmp.addr = addr_M[PEid];
                payload_out_tmp.incr = incr_M[PEid];
                output_payloads[PEid].write(payload_out_tmp);
                process_cnt_incr++;
            }
        }
        if (!prev_finish) {
            prev_finish = num_payloads_in.read_nb(num_payload);
        }
        process_cnt += process_cnt_incr;
        bool process_complete = (process_cnt == num_payload);
        loop_exit = prev_finish && process_complete;
        // ----- end of M stage

        // sw_emu line tracing
        // #ifndef __SYNTHESIS__
        // if (line_tracing_ufixed_pe_cluster) {
        //     std::cout << "INFO: [kernel SpMSpV (ufixed) part1] loop count: " << sw_emu_iter_cnt_p1 << "["
        //               << (prev_finish ? "PF|" : "..|") << (process_complete ? "PC]" : "..]")
        //               << "[" << process_cnt << " / " << num_payload << "]" << std::endl << std::flush;
        //     for (unsigned i = 0; i < num_PE; i++) {
        //         std::cout << "  PE[" << i << "]"
        //                   << (valid_F[i] ? "F " : ". ")
        //                   << (valid_M[i] ? "M " : ". ") << std::endl << std::flush;
        //     }
        // }
        // sw_emu_iter_cnt_p1++;
        // if (sw_emu_early_abort && sw_emu_iter_cnt_p1 > sw_emu_iter_limit) {
        //     std::cout << "ERROR: [kernel SpMSpV (ufixed) part1] sw_emu iteration limit("
        //               << sw_emu_iter_limit << ") exceeded!" << std::endl << std::flush;
        //     std::cout << "  Aborting!" << std::endl << std::flush;
        //     return;
        // }
        // #endif
    }
    num_payloads_out.write(num_payload);
}

// ufixed pe cluster part 2 for SpMSpV using uram
template<typename ValT, typename OpT, unsigned num_hbm_channels, unsigned num_PE, unsigned bank_size, unsigned addr_shamt>
void ufixed_pe_cluster_spmspv_uram_part2(
    hls::stream<PP_STREAM_T<ValT> > input_payloads[num_PE],
    hls::stream<unsigned> &num_payloads_in,
    ValT output_buffer[num_hbm_channels][num_PE][bank_size],
    OpT Op,
    ValT Zero
) {
    // loop control
    bool prev_finish = false;
    unsigned num_payload = 0;
    unsigned process_cnt = 0;
    bool loop_exit = false;

    #ifndef __SYNTHESIS__
    int sw_emu_iter_cnt_p2 = 0;
    #endif

    // F stage (depth 1)
    unsigned addr_S[num_PE];
    #pragma HLS array_partition variable=addr_S complete
    ValT incr_S[num_PE];
    #pragma HLS array_partition variable=incr_S complete
    bool valid_S[num_PE];
    #pragma HLS array_partition variable=valid_S complete
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        addr_S[i] = 0;
        incr_S[i] = 0;
        valid_S[i] = false;
    }

    // R stage (depth 2)
    ValT q_R[num_PE];
    #pragma HLS array_partition variable=q_R complete
    ValT incr_R[num_PE];
    #pragma HLS array_partition variable=incr_R complete
    unsigned addr_R[num_PE];
    #pragma HLS array_partition variable=addr_R complete
    bool valid_R[num_PE];
    #pragma HLS array_partition variable=valid_R complete
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        q_R[i] = 0;
        incr_R[i] = 0;
        addr_R[i] = 0;
        valid_R[i] = false;
    }

    // D stage (depth 1)
    ValT q_D[num_PE];
    #pragma HLS array_partition variable=q_D complete
    ValT incr_D[num_PE];
    #pragma HLS array_partition variable=incr_D complete
    unsigned addr_D[num_PE];
    #pragma HLS array_partition variable=addr_D complete
    bool valid_D[num_PE];
    #pragma HLS array_partition variable=valid_D complete
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        q_D[i] = 0;
        incr_D[i] = 0;
        addr_D[i] = 0;
        valid_D[i] = false;
    }

    // A stage (depth 1)
    ValT q_A[num_PE];
    #pragma HLS array_partition variable=q_A complete
    ValT incr_A[num_PE];
    #pragma HLS array_partition variable=incr_A complete
    ValT new_q_A[num_PE];
    #pragma HLS array_partition variable=new_q_A complete
    unsigned addr_A[num_PE];
    #pragma HLS array_partition variable=addr_A complete
    bool valid_A[num_PE];
    #pragma HLS array_partition variable=valid_A complete
    unsigned in_flight_A1_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_A1_addr complete
    ValT in_flight_A1_data[num_PE];
    #pragma HLS array_partition variable=in_flight_A1_data complete
    bool in_flight_A1_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_A1_valid complete

    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        q_A[i] = 0;
        incr_A[i] = 0;
        new_q_A[i] = 0;
        addr_A[i] = 0;
        valid_A[i] = false;
        in_flight_A1_addr[i] = 0;
        in_flight_A1_data[i] = 0;
        in_flight_A1_valid[i] = false;
    }

    // W stage (depth 2)
    ValT new_q_W[num_PE];
    #pragma HLS array_partition variable=new_q_W complete
    unsigned addr_W[num_PE];
    #pragma HLS array_partition variable=addr_W complete
    bool valid_W[num_PE];
    #pragma HLS array_partition variable=valid_W complete
    unsigned in_flight_W1_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_W1_addr complete
    unsigned in_flight_W2_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_W2_addr complete
    ValT in_flight_W1_data[num_PE];
    #pragma HLS array_partition variable=in_flight_W1_data complete
    ValT in_flight_W2_data[num_PE];
    #pragma HLS array_partition variable=in_flight_W2_data complete
    bool in_flight_W1_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_W1_valid complete
    bool in_flight_W2_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_W2_valid complete
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        new_q_W[i] = 0;
        addr_W[i] = 0;
        valid_W[i] = false;
        in_flight_W1_addr[i] = 0;
        in_flight_W2_addr[i] = 0;
        in_flight_W1_data[i] = 0;
        in_flight_W2_data[i] = 0;
        in_flight_W1_valid[i] = false;
        in_flight_W2_valid[i] = false;
    }

    // pipeline control signals
    bool match_AW1[num_PE];
    #pragma HLS array_partition variable=match_AW1 complete
    bool match_AW2[num_PE];
    #pragma HLS array_partition variable=match_AW2 complete
    bool match_AA1[num_PE];
    #pragma HLS array_partition variable=match_AA1 complete
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        match_AW1[i] = false;
        match_AW2[i] = false;
        match_AA1[i] = false;
    }

    loop_pe_part2:
    while (!loop_exit) {
        #pragma HLS pipeline II=1
        #pragma HLS latency min=6 max=6
        #pragma HLS dependence variable=output_buffer inter RAW false
        #pragma HLS dependence variable=loop_exit inter RAW true distance=7
        #pragma HLS dependence variable=in_flight_A1_addr inter RAW true distance=1
        #pragma HLS dependence variable=in_flight_W1_addr inter RAW true distance=2
        #pragma HLS dependence variable=in_flight_W2_addr inter RAW true distance=3
        #pragma HLS dependence variable=in_flight_A1_data inter RAW true distance=1
        #pragma HLS dependence variable=in_flight_W1_data inter RAW true distance=2
        #pragma HLS dependence variable=in_flight_W2_data inter RAW true distance=3
        #pragma HLS dependence variable=in_flight_A1_valid inter RAW true distance=1
        #pragma HLS dependence variable=in_flight_W1_valid inter RAW true distance=2
        #pragma HLS dependence variable=in_flight_W2_valid inter RAW true distance=3

        // Fetch stage (F)
        loop_F:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            PP_STREAM_T<ValT> payload_in_tmp;
            valid_S[PEid] = input_payloads[PEid].read_nb(payload_in_tmp);
            if (valid_S[PEid]) {
                addr_S[PEid] = payload_in_tmp.addr;
                incr_S[PEid] = payload_in_tmp.incr;
            } else {
                addr_S[PEid] = 0;
                incr_S[PEid] = 0;
            }
        }
        // ----- end of F stage

        // Read stage (R)
        loop_R:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            valid_R[PEid] = valid_S[PEid];
            addr_R[PEid] = addr_S[PEid];
            incr_R[PEid] = incr_S[PEid];
            if (valid_R[PEid]) {
                q_R[PEid] = output_buffer[(addr_R[PEid] >> addr_shamt) % num_hbm_channels][PEid]
                                         [(addr_R[PEid] >> addr_shamt) / num_hbm_channels];
            } else {
                q_R[PEid] = 0;
            }
        }
        // ----- end of R stage

        // Decode stage (D) - additional stage for data forwarding
        loop_ex:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            valid_D[PEid] = valid_R[PEid];
            addr_D[PEid] = addr_R[PEid];
            incr_D[PEid] = incr_R[PEid];
            bool match_A1 = ((addr_D[PEid] == in_flight_A1_addr[PEid]) && valid_R[PEid] && in_flight_A1_valid[PEid]);
            bool match_W1 = ((addr_D[PEid] == in_flight_W1_addr[PEid]) && valid_R[PEid] && in_flight_W1_valid[PEid]);
            bool match_W2 = ((addr_D[PEid] == in_flight_W2_addr[PEid]) && valid_R[PEid] && in_flight_W2_valid[PEid]);
            if (match_A1) {
                q_D[PEid] = in_flight_A1_data[PEid];
            } else if (match_W1) {
                q_D[PEid] = in_flight_W1_data[PEid];
            } else if (match_W2) {
                q_D[PEid] = in_flight_W2_data[PEid];
            } else {
                q_D[PEid] = q_R[PEid];
            }
        }
        // ----- end of D stage

        // Add stage (A)
        loop_A:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            addr_A[PEid] = addr_R[PEid];
            valid_A[PEid] = valid_R[PEid];
            incr_A[PEid] = incr_R[PEid];
            q_A[PEid] = q_D[PEid];
            new_q_A[PEid] = pe_ufixed_add_alu<ValT, OpT>(q_A[PEid], incr_A[PEid], Zero, Op, valid_A[PEid]);
        }
        // ----- end of A stage

        // update in-flight registers
        loop_update_in_flight:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            in_flight_W2_valid[PEid] = in_flight_W1_valid[PEid];
            in_flight_W1_valid[PEid] = in_flight_A1_valid[PEid];
            in_flight_A1_valid[PEid] = valid_A[PEid];

            in_flight_W2_addr[PEid] = in_flight_W1_addr[PEid];
            in_flight_W1_addr[PEid] = in_flight_A1_addr[PEid];
            in_flight_A1_addr[PEid] = addr_A[PEid];

            in_flight_W2_data[PEid] = in_flight_W1_data[PEid];
            in_flight_W1_data[PEid] = in_flight_A1_data[PEid];
            in_flight_A1_data[PEid] = new_q_A[PEid];
        }

        // Write stage (W)
        unsigned process_cnt_incr = 0;
        loop_W:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            addr_W[PEid] = addr_A[PEid];
            valid_W[PEid] = valid_A[PEid];
            new_q_W[PEid] = new_q_A[PEid];
            if (valid_W[PEid]) {
                output_buffer[(addr_W[PEid] >> addr_shamt) % num_hbm_channels][PEid]
                             [(addr_W[PEid] >> addr_shamt) / num_hbm_channels] = new_q_W[PEid];
                process_cnt_incr++;
            }
        }
        if (!prev_finish) {
            prev_finish = num_payloads_in.read_nb(num_payload);
        }
        process_cnt += process_cnt_incr;
        bool process_complete = (process_cnt == num_payload);
        loop_exit = process_complete && prev_finish;

        // sw_emu line tracing
        // #ifndef __SYNTHESIS__
        // if (line_tracing_ufixed_pe_cluster) {
        //     std::cout << "INFO: [kernel SpMSpV (ufixed) part2] loop count: "
        //               << sw_emu_iter_cnt_p2 << std::endl << std::flush;
        // }
        // sw_emu_iter_cnt_p2++;
        // if (sw_emu_early_abort && sw_emu_iter_cnt_p2 > sw_emu_iter_limit) {
        //     std::cout << "ERROR: [kernel SpMSpV (ufixed) part2] sw_emu iteration limit("
        //               << sw_emu_iter_limit << ") exceeded!" << std::endl << std::flush;
        //     std::cout << "  Aborting!" << std::endl << std::flush;
        //     return;
        // }
        // #endif
    }
}


// whole pe cluster for SpMSpV using uram
template<typename ValT, typename OpT, typename PayloadT,
         unsigned num_hbm_channels, unsigned num_PE, unsigned addr_shamt, unsigned bank_size>
void ufixed_pe_cluster_spmspv_uram(
    hls::stream<PayloadT> input_payloads[num_PE],
    ValT output_buffer[num_hbm_channels][num_PE][bank_size],
    OpT Op,
    ValT Zero,
    hls::stream<unsigned> &num_payloads_in
) {
    #pragma HLS dataflow

    hls::stream<PP_STREAM_T<ValT> > PP_stream[num_PE];
    #pragma HLS stream variable=PP_stream depth=4
    hls::stream<unsigned> PP_npld_stream;
    #pragma HLS stream variable=PP_npld_stream depth=2

    ufixed_pe_cluster_part1<ValT, OpT, PayloadT, num_PE>(input_payloads,
        PP_stream, num_payloads_in, PP_npld_stream, Op, Zero);
    ufixed_pe_cluster_spmspv_uram_part2<ValT, OpT, num_hbm_channels, num_PE, bank_size, addr_shamt>(PP_stream,
        PP_npld_stream, output_buffer, Op, Zero);
}

#endif  // GRAPHLILY_HW_UFIXED_PE_H_
