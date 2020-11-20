#ifndef GRAPHLILY_HW_FLOAT_PE_H_
#define GRAPHLILY_HW_FLOAT_PE_H_

#include "hls_stream.h"
#include <iostream>
#include <iomanip>

#include "./math_constants.h"
#include "./util.h"

#ifndef __SYNTHESIS__
bool line_tracing_float_pe_cluster = false;
bool sw_emu_early_abort = false;
unsigned sw_emu_iter_limit = 100;
#endif

#define MIN(a, b) ((a < b)? a : b)

//----------------------------------------------------------------
// ALUs
//----------------------------------------------------------------

// floating-point saturation addition
float float_sat_add(float a, float b) {
    // #pragma HLS inline
    #pragma HLS pipeline II=1
    #pragma HLS latency min=5 max=5
    float x = a + b;
    if (a >= FLTINF) return FLTINF;
    if (b >= FLTINF) return FLTINF;
    if (x >= FLTINF) return FLTINF;
    return x;
}

template<typename OpT>
float pe_float_mul_alu(float a, float b, float z, OpT op, bool en) {
    #pragma HLS pipeline II=1
    #pragma HLS latency min=7 max=7
    float out;
    switch (op) {
        case MULADD:
            out = a * b;
            break;
        case ANDOR:
            out = a && b;
            break;
        case ADDMIN:
            out = float_sat_add(a, b);
            break;
        default:
            out = z;  // TODO: what is z?
                      // -- z is the zero value in this semiring
            break;
    }
    return en ? out : z;
}


template<typename OpT>
float pe_float_add_alu(float a, float b, float z, OpT op, bool en) {
    #pragma HLS pipeline II=1
    #pragma HLS latency min=5 max=5
    float out;
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
// float_pe_cluster
//----------------------------------------------------------------

/*
  The floating-point PE is divided into 2 parts. These two parts are connected via a FIFO.

  part1: fetch the payload from previous stage, and do the <x> operator in the semiring.
         this part has no data dependencies.
  part2: read the current value from the output buffer, do the <+> operator and wirte back to the output buffer.
         this part has data dependencies and the largest distance is 9.

  If we merge these two parts, we still need a skid buffer since it is not possible to stall the floating-point multiplier
  manully in HLS. We have to use a FIFO to apply back pressure and the tool will implement the stall logic.
*/

typedef struct P1_to_P2_stream_entry_type {
    unsigned addr;
    float incr;
} PP_STREAM_T;

// float pe cluster part 1:
template<typename OpT, typename PayloadT,
         unsigned num_PE>
void float_pe_cluster_part1(
    hls::stream<PayloadT> input_payloads[num_PE],
    hls::stream<PP_STREAM_T> output_payloads[num_PE],
    hls::stream<unsigned> &num_payloads_in,
    hls::stream<unsigned> &num_payloads_out,
    OpT Op,
    float Zero
) {
    // loop control
    bool prev_finish = false;
    unsigned num_payload = 0;
    unsigned process_cnt = 0;
    bool process_complete = false;

    #ifndef __SYNTHESIS__
    int sw_emu_iter_cnt_p1 = 0;
    #endif

    // pipeline registers
    // F stage (depth 1)
    float a_F[num_PE];
    #pragma HLS array_partition variable=a_F complete
    float b_F[num_PE];
    #pragma HLS array_partition variable=b_F complete
    unsigned addr_F[num_PE];
    #pragma HLS array_partition variable=addr_F complete
    bool  valid_F[num_PE];
    #pragma HLS array_partition variable=valid_F complete
    for (unsigned i = 0; i < num_PE; i++)  {
        #pragma HLS unroll
        a_F[i] = 0;
        b_F[i] = 0;
        addr_F[i] = 0;
        valid_F[i] = false;
    }

    // M stage (depth 8)
    float incr_M[num_PE];
    #pragma HLS array_partition variable=incr_M complete
    unsigned addr_M[num_PE];
    #pragma HLS array_partition variable=addr_M complete
    bool valid_M[num_PE];
    #pragma HLS array_partition variable=valid_M complete
    for (unsigned i = 0; i < num_PE; i++)  {
        #pragma HLS unroll
        incr_M[i] = 0;
        addr_M[i] = 0;
        valid_M[i] = false;
    }

    loop_pe_part1:
    while ( !(prev_finish && process_complete) ) {
        #pragma HLS pipeline II=1
        #pragma HLS dependence variable=process_complete inter RAW true distance=10

        // Fetch stage (F)
        if(!prev_finish) {
            prev_finish = num_payloads_in.read_nb(num_payload);
        }

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
        loop_M:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            valid_M[PEid] = valid_F[PEid];
            addr_M[PEid] = addr_F[PEid];
            incr_M[PEid] = pe_float_mul_alu<OpT>(a_F[PEid], b_F[PEid], Zero, Op, valid_M[PEid]);
            if (valid_M[PEid]) {
                PP_STREAM_T payload_out_tmp;
                payload_out_tmp.addr = addr_M[PEid];
                payload_out_tmp.incr = incr_M[PEid];
                output_payloads[PEid].write(payload_out_tmp);
            }
        }
        process_cnt += array_popcount<num_PE>(valid_M);
        process_complete = (process_cnt == num_payload);
        // ----- end of M stage

        // sw_emu line tracing
        #ifndef __SYNTHESIS__
        if(line_tracing_float_pe_cluster) {
            std::cout << "INFO: [kernel SpMSpV (float) part1] loop count: " << sw_emu_iter_cnt_p1 << "["
                      << (prev_finish ? "PF|" : "..|") << (process_complete ? "PC]" : "..]")
                      << "[" << process_cnt << " / " << num_payload << "]" << std::endl << std::flush;
            for (unsigned i = 0; i < num_PE; i++) {
                std::cout << "  PE[" << i << "]"
                          << (valid_F[i] ? "F " : ". ")
                          << (valid_M[i] ? "M " : ". ") << std::endl << std::flush;
            }

        }
        sw_emu_iter_cnt_p1 ++;
        if (sw_emu_early_abort && sw_emu_iter_cnt_p1 > sw_emu_iter_limit) {
            std::cout << "ERROR: [kernel SpMSpV (float) part1] sw_emu iteration limit(" << sw_emu_iter_limit << ") exceeded!" << std::endl << std::flush;
            std::cout << "  Aborting!" << std::endl << std::flush;
            return;
        }
        #endif
    }
    num_payloads_out.write(num_payload);

}

// float pe cluster part 2:
template<typename OpT,
         unsigned num_PE, unsigned bank_size, unsigned addr_shamt>
void float_pe_cluster_part2(
    hls::stream<PP_STREAM_T> input_payloads[num_PE],
    hls::stream<unsigned> &num_payloads_in,
    float output_buffer[num_PE][bank_size],
    OpT Op,
    float Zero
) {
    // loop control
    bool prev_finish = false;
    unsigned num_payload = 0;
    unsigned process_cnt = 0;
    bool process_complete = false;

    #ifndef __SYNTHESIS__
    int sw_emu_iter_cnt_p2 = 0;
    #endif

    // S stage (depth 1)
    unsigned addr_S[num_PE];
    #pragma HLS array_partition variable=addr_S complete
    float incr_S[num_PE];
    #pragma HLS array_partition variable=incr_S complete
    bool valid_S[num_PE];
    #pragma HLS array_partition variable=valid_S complete
    bool next_valid_R[num_PE];
    #pragma HLS array_partition variable=next_valid_R complete
    for (unsigned i = 0; i < num_PE; i++)  {
        #pragma HLS unroll
        addr_S[i] = 0;
        incr_S[i] = 0;
        valid_S[i] = false;
        next_valid_R[i] = false;
    }

    // R stage (depth 1)
    float q_R[num_PE];
    #pragma HLS array_partition variable=q_R complete
    float incr_R[num_PE];
    #pragma HLS array_partition variable=incr_R complete
    unsigned addr_R[num_PE];
    #pragma HLS array_partition variable=addr_R complete
    bool valid_R[num_PE];
    #pragma HLS array_partition variable=valid_R complete
    unsigned in_flight_R1_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_R1_addr complete
    bool in_flight_R1_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_R1_valid complete
    for (unsigned i = 0; i < num_PE; i++)  {
        #pragma HLS unroll
        q_R[i] = 0;
        incr_R[i] = 0;
        addr_R[i] = 0;
        valid_R[i] = false;
        in_flight_R1_addr[i] = 0;
        in_flight_R1_valid[i] = false;
    }

    // A stage (depth 6)
    float q_A[num_PE];
    #pragma HLS array_partition variable=q_A complete
    float incr_A[num_PE];
    #pragma HLS array_partition variable=incr_A complete
    float new_q_A[num_PE];
    #pragma HLS array_partition variable=new_q_A complete
    unsigned addr_A[num_PE];
    #pragma HLS array_partition variable=addr_A complete
    bool valid_A[num_PE];
    #pragma HLS array_partition variable=valid_A complete
    unsigned in_flight_A1_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_A1_addr complete
    unsigned in_flight_A2_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_A2_addr complete
    unsigned in_flight_A3_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_A3_addr complete
    unsigned in_flight_A4_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_A4_addr complete
    unsigned in_flight_A5_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_A5_addr complete
    unsigned in_flight_A6_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_A6_addr complete
    bool in_flight_A1_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_A1_valid complete
    bool in_flight_A2_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_A2_valid complete
    bool in_flight_A3_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_A3_valid complete
    bool in_flight_A4_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_A4_valid complete
    bool in_flight_A5_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_A5_valid complete
    bool in_flight_A6_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_A6_valid complete
    for (unsigned i = 0; i < num_PE; i++)  {
        #pragma HLS unroll
        q_A[i] = 0;
        incr_A[i] = 0;
        new_q_A[i] = 0;
        addr_A[i] = 0;
        valid_A[i] = false;
        in_flight_A1_addr[i] = 0;
        in_flight_A2_addr[i] = 0;
        in_flight_A3_addr[i] = 0;
        in_flight_A4_addr[i] = 0;
        in_flight_A5_addr[i] = 0;
        in_flight_A6_addr[i] = 0;
        in_flight_A1_valid[i] = false;
        in_flight_A2_valid[i] = false;
        in_flight_A3_valid[i] = false;
        in_flight_A4_valid[i] = false;
        in_flight_A5_valid[i] = false;
        in_flight_A6_valid[i] = false;
    }

    // W stage (depth 2)
    float new_q_W[num_PE];
    #pragma HLS array_partition variable=new_q_W complete
    unsigned addr_W[num_PE];
    #pragma HLS array_partition variable=addr_W complete
    bool valid_W[num_PE];
    #pragma HLS array_partition variable=valid_W complete
    unsigned in_flight_W1_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_W1_addr complete
    unsigned in_flight_W2_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_W2_addr complete
    bool in_flight_W1_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_W1_valid complete
    bool in_flight_W2_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_W2_valid complete
    for (unsigned i = 0; i < num_PE; i++)  {
        #pragma HLS unroll
        new_q_W[i] = 0;
        addr_W[i] = 0;
        valid_W[i] = false;
        in_flight_W1_addr[i] = 0;
        in_flight_W2_addr[i] = 0;
        in_flight_W1_valid[i] = false;
        in_flight_W2_valid[i] = false;
    }

    // pipeline control signals
    bool match_SR[num_PE];
    #pragma HLS array_partition variable=match_SR complete
    bool match_SA[num_PE];
    #pragma HLS array_partition variable=match_SA complete
    bool match_SW[num_PE];
    #pragma HLS array_partition variable=match_SW complete
    bool stall_S[num_PE];
    #pragma HLS array_partition variable=stall_S complete
    for (unsigned i = 0; i < num_PE; i++)  {
        #pragma HLS unroll
        match_SR[i] = false;
        match_SA[i] = false;
        match_SW[i] = false;
        stall_S[i] = false;
    }

    loop_pe_part2:
    while ( !(prev_finish && process_complete) ) {
        #pragma HLS pipeline II=1
        #pragma HLS dependence variable=output_buffer inter RAW false
        #pragma HLS dependence variable=process_complete inter RAW true distance=10

        if(!prev_finish) {
            prev_finish = num_payloads_in.read_nb(num_payload);
        }

        // Skid stage (S)
        loop_S:
        for (unsigned PEid = 0; PEid < num_PE; PEid++)  {
            #pragma HLS unroll
            PP_STREAM_T payload_in_tmp;
            if (!stall_S[PEid]) {
                valid_S[PEid] = input_payloads[PEid].read_nb(payload_in_tmp);
                if (valid_S[PEid]) {
                    addr_S[PEid] = payload_in_tmp.addr;
                    incr_S[PEid] = payload_in_tmp.incr;
                } else {
                    addr_S[PEid] = 0;
                    incr_S[PEid] = 0;
                }
            } else {
                valid_S[PEid] = true;
                addr_S[PEid] = addr_S[PEid];
                incr_S[PEid] = incr_S[PEid];
            }

            match_SR[PEid] = ((addr_S[PEid] == in_flight_R1_addr[PEid]) && valid_S[PEid] && in_flight_R1_valid[PEid]);

            match_SA[PEid] = ((addr_S[PEid] == in_flight_A1_addr[PEid]) && valid_S[PEid] && in_flight_A1_valid[PEid])
                          || ((addr_S[PEid] == in_flight_A2_addr[PEid]) && valid_S[PEid] && in_flight_A2_valid[PEid])
                          || ((addr_S[PEid] == in_flight_A3_addr[PEid]) && valid_S[PEid] && in_flight_A3_valid[PEid])
                          || ((addr_S[PEid] == in_flight_A4_addr[PEid]) && valid_S[PEid] && in_flight_A4_valid[PEid])
                          || ((addr_S[PEid] == in_flight_A5_addr[PEid]) && valid_S[PEid] && in_flight_A5_valid[PEid])
                          || ((addr_S[PEid] == in_flight_A6_addr[PEid]) && valid_S[PEid] && in_flight_A6_valid[PEid]);

            match_SW[PEid] = ((addr_S[PEid] == in_flight_W1_addr[PEid]) && valid_S[PEid] && in_flight_W1_valid[PEid])
                          || ((addr_S[PEid] == in_flight_W2_addr[PEid]) && valid_S[PEid] && in_flight_W2_valid[PEid]);

            stall_S[PEid] = match_SR[PEid] || match_SA[PEid] || match_SW[PEid];
            next_valid_R[PEid] = valid_S[PEid] && !stall_S[PEid];
        }
        // ----- end of S stage

        // update in-flight registers
        loop_update_in_flight:
        for (unsigned PEid = 0; PEid < num_PE; PEid++)  {
            #pragma HLS unroll
            in_flight_W2_valid[PEid] = in_flight_W1_valid[PEid];
            in_flight_W1_valid[PEid] = in_flight_A6_valid[PEid];
            in_flight_A6_valid[PEid] = in_flight_A5_valid[PEid];
            in_flight_A5_valid[PEid] = in_flight_A4_valid[PEid];
            in_flight_A4_valid[PEid] = in_flight_A3_valid[PEid];
            in_flight_A3_valid[PEid] = in_flight_A2_valid[PEid];
            in_flight_A2_valid[PEid] = in_flight_A1_valid[PEid];
            in_flight_A1_valid[PEid] = in_flight_R1_valid[PEid];
            in_flight_R1_valid[PEid] = next_valid_R[PEid];

            in_flight_W2_addr[PEid] = in_flight_W1_addr[PEid];
            in_flight_W1_addr[PEid] = in_flight_A6_addr[PEid];
            in_flight_A6_addr[PEid] = in_flight_A5_addr[PEid];
            in_flight_A5_addr[PEid] = in_flight_A4_addr[PEid];
            in_flight_A4_addr[PEid] = in_flight_A3_addr[PEid];
            in_flight_A3_addr[PEid] = in_flight_A2_addr[PEid];
            in_flight_A2_addr[PEid] = in_flight_A1_addr[PEid];
            in_flight_A1_addr[PEid] = in_flight_R1_addr[PEid];
            in_flight_R1_addr[PEid] = addr_S[PEid];
        }

        // Read stage (R)
        loop_R:
        for (unsigned PEid = 0; PEid < num_PE; PEid++)  {
            #pragma HLS unroll
            valid_R[PEid] = next_valid_R[PEid];
            addr_R[PEid] = addr_S[PEid];
            incr_R[PEid] = incr_S[PEid];
            if (valid_R[PEid]) {
                q_R[PEid] = output_buffer[PEid][addr_R[PEid] >> addr_shamt];
            } else {
                q_R[PEid] = 0;
            }
        }
        // ----- end of R stage

        // Add stage (A)
        loop_A:
        for (unsigned PEid = 0; PEid < num_PE; PEid++)  {
            #pragma HLS unroll
            addr_A[PEid] = addr_R[PEid];
            valid_A[PEid] = valid_R[PEid];
            q_A[PEid] = q_R[PEid];
            incr_A[PEid] = incr_R[PEid];
            new_q_A[PEid] = pe_float_add_alu<OpT>(q_A[PEid], incr_A[PEid], Zero, Op, valid_A[PEid]);
        }
        // ----- end of A stage

        // Write stage (W)
        loop_W:
        for (unsigned PEid = 0; PEid < num_PE; PEid++)  {
            #pragma HLS unroll
            addr_W[PEid] = addr_A[PEid];
            valid_W[PEid] = valid_A[PEid];
            new_q_W[PEid] = new_q_A[PEid];
            if (valid_W[PEid]) {
                output_buffer[PEid][addr_W[PEid] >> addr_shamt] = new_q_W[PEid];

            }
        }
        process_cnt += array_popcount<num_PE>(valid_W);
        process_complete = (process_cnt == num_payload);

        // sw_emu line tracing
        #ifndef __SYNTHESIS__
        if(line_tracing_float_pe_cluster) {
            std::cout << "INFO: [kernel SpMSpV (float) part2] loop count: " << sw_emu_iter_cnt_p2 << std::endl << std::flush;
        }
        sw_emu_iter_cnt_p2 ++;
        if (sw_emu_early_abort && sw_emu_iter_cnt_p2 > sw_emu_iter_limit) {
            std::cout << "ERROR: [kernel SpMSpV (float) part2] sw_emu iteration limit(" << sw_emu_iter_limit << ") exceeded!" << std::endl << std::flush;
            std::cout << "  Aborting!" << std::endl << std::flush;
            return;
        }
        #endif

    }
}

// whole pe cluster
template<typename ValT, typename OpT, typename PayloadT,
         unsigned num_PE, unsigned addr_shamt, unsigned bank_size>
void float_pe_cluster(
    hls::stream<PayloadT> input_payloads[num_PE],
    ValT output_buffer[num_PE][bank_size],
    OpT Op,
    ValT Zero,
    hls::stream<unsigned> &num_payloads_in
) {
    #pragma HLS dataflow

    hls::stream<PP_STREAM_T> PP_stream[num_PE];
    #pragma HLS stream variable=PP_stream depth=16
    hls::stream<unsigned> PP_npld_stream;
    #pragma HLS stream variable=PP_npld_stream depth=2

    float_pe_cluster_part1<OpT, PayloadT, num_PE>(input_payloads, PP_stream, num_payloads_in, PP_npld_stream, Op, Zero);
    float_pe_cluster_part2<OpT, num_PE, bank_size, addr_shamt>(PP_stream, PP_npld_stream, output_buffer, Op, Zero);
}

#endif  // GRAPHLILY_HW_FLOAT_PE_H_
