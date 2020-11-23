#ifndef GRAPHLILY_HW_SHUFFLE_H_
#define GRAPHLILY_HW_SHUFFLE_H_

#include <iostream>
#include <iomanip>

#include "hls_stream.h"

#include "./util.h"


#ifndef __SYNTHESIS__
bool line_tracing_shuffle_1p = false;
#endif

// pipeline register
// Its latency is 5.
// Only use this function to wrap signals that travels along stage A!
template<typename PayloadT>
PayloadT pipereg_stage_A (PayloadT in) {
    #pragma HLS pipeline II=1
    #pragma HLS latency min=5 max=5
    return in;
}

//------------------------------------------------------------
// arbiters
//------------------------------------------------------------

// TODO: Do we really need to expose addr_mask as an argument? Can it be inferred?
// TODO: Is num_in_lane always equal to num_out_lane?
template<unsigned num_in_lane, unsigned num_out_lane, unsigned addr_mask>
unsigned arbiter_1p(
    unsigned in_addr[num_in_lane],
    bool     in_valid[num_in_lane],
    // bool     in_granted[num_in_lane],
    bool     in_resend[num_in_lane],
    unsigned xbar_sel[num_out_lane],
    bool     out_valid[num_out_lane],
    unsigned rotate_priority
) {
    #pragma HLS pipeline II=1
    #pragma HLS latency min=5 max=5
    // #pragma HLS inline

    // bool in_granted[num_in_lane];
    // #pragma HLS array_partition variable=in_granted complete

    // static unsigned in_addr[num_in_lane];
    // #pragma HLS array_partition variable=in_addr complete
    // loop_A_extract_addr:
    // for (unsigned ILid = 0; ILid < num_in_lane; ILid++) {
    //     in_addr[ILid] = in_payload[ILid].index;
    // }

    // prioritized valid and addr
    bool arb_p_in_valid[num_in_lane];
    #pragma HLS array_partition variable=arb_p_in_valid complete
    unsigned arb_p_in_addr[num_in_lane];
    #pragma HLS array_partition variable=arb_p_in_addr complete

    array_shift_left<unsigned, num_in_lane>(in_addr, arb_p_in_addr, rotate_priority);
    array_shift_left<bool, num_in_lane>(in_valid, arb_p_in_valid, rotate_priority);

    loop_A_arbsearch:
    for (unsigned OLid = 0; OLid < num_out_lane; OLid++) {
        #pragma HLS unroll
        bool found = false;
        unsigned chosen_port = 0;

        loop_ab_logic_encoder_unroll:
        for (unsigned ILid_plus_1 = num_in_lane; ILid_plus_1 > 0; ILid_plus_1--) {
            #pragma HLS unroll
            if (arb_p_in_valid[ILid_plus_1 - 1] && ((arb_p_in_addr[ILid_plus_1 - 1] & addr_mask) == OLid)) {
                chosen_port = ILid_plus_1 - 1;
                found = true;
            }
        }
        if (!found) {
            out_valid[OLid] = false;
            xbar_sel[OLid] = 0;
        } else {
            out_valid[OLid] = true;
            xbar_sel[OLid] = chosen_port;
        }
    }

    array_cyclic_add<unsigned,num_out_lane,num_in_lane>(xbar_sel, out_valid, rotate_priority);

    unsigned grant_count = 0;
    loop_A_grant:
    for (unsigned ILid = 0; ILid < num_in_lane; ILid++) {
        #pragma HLS unroll
        unsigned requested_olid = in_addr[ILid] & addr_mask;
        bool in_granted = (in_valid[ILid]
                           && out_valid[requested_olid]
                           && (xbar_sel[requested_olid] == ILid));
        in_resend[ILid] = in_valid[ILid] && !in_granted;
        if (in_granted) grant_count++;
    }

    // loop_A_resend:
    // for (unsigned ILid = 0; ILid < num_in_lane; ILid++) {
    //     #pragma HLS unroll
    //     // resend path
    //     in_resend[ILid] = in_valid[ILid] && !in_granted[ILid];
    // }

    // loop_A_pass:
    // for (unsigned ILid = 0; ILid < num_in_lane; ILid++) {
    //     out_payload[ILid] = in_payload[ILid];
    // }

    return grant_count;
}


/* shuffler-1p
 * 1 downstream entitiy could process 1 payload per cycle
 * type PayloadT must be a struct of 2 fields:
 * unsigned index
 * <PayloadValT> data
*/
template<typename PayloadT, typename PayloadValT,
         unsigned num_in_lane, unsigned num_out_lane, unsigned addr_mask>
void shuffler_1p(
    // fifos
    hls::stream<PayloadT> input_lanes[num_in_lane],
    hls::stream<PayloadT> output_lanes[num_out_lane],
    // total number of payloads (avaliable after all payloads are put into the input lanes)
    hls::stream<unsigned> &num_payloads_in,
    // all outputs are in the output lanes
    hls::stream<unsigned> &num_payloads_out
) {
    // pipeline control variables
    bool prev_finish = false;
    unsigned payload_cnt = 0;
    unsigned num_granted_A = 0;
    unsigned num_granted_C = 0;
    unsigned process_cnt = 0;

    // pipeline data registers before arbiter
    PayloadT payload_F[num_in_lane];
    PayloadT payload_A[num_in_lane];
    unsigned addr_A[num_in_lane];
    #pragma HLS array_partition variable=payload_F complete
    #pragma HLS array_partition variable=payload_A complete
    #pragma HLS array_partition variable=addr_A complete

    // pipeline data registers after arbiter
    PayloadT payload_C[num_in_lane];
    #pragma HLS array_partition variable=payload_C complete

    // pipeline valid registers before arbiter
    bool valid_F[num_in_lane];
    bool valid_A[num_in_lane];
    #pragma HLS array_partition variable=valid_F complete
    #pragma HLS array_partition variable=valid_A complete

    // pipeline valid registers after arbiter
    unsigned xbar_sel_C[num_out_lane];
    bool xbar_valid_C[num_out_lane];
    bool valid_C[num_in_lane];
    #pragma HLS array_partition variable=xbar_sel_C complete
    #pragma HLS array_partition variable=xbar_valid_C complete
    #pragma HLS array_partition variable=valid_C complete

    // resend control
    PayloadT payload_resend[num_in_lane];
    bool resend[num_in_lane];
    #pragma HLS data_pack variable=payload_resend
    #pragma HLS array_partition variable=payload_resend complete
    #pragma HLS array_partition variable=resend complete

    // loop control
    bool loop_exit = false;

    // arbiter inputs
    unsigned arbiter_in_addr[num_in_lane];
    bool arbiter_in_valid[num_in_lane];
    #pragma HLS array_partition variable=arbiter_in_addr complete
    #pragma HLS array_partition variable=arbiter_in_valid complete

    // arbiter outputs
    // bool arbiter_in_granted[num_in_lane];
    unsigned xbar_sel_A[num_out_lane];
    bool xbar_valid_A[num_out_lane];
    // #pragma HLS array_partition variable=arbiter_in_granted complete
    #pragma HLS array_partition variable=xbar_sel_A complete
    #pragma HLS array_partition variable=xbar_valid_A complete

    // arbiter priority rotation
    unsigned rotate_priority = 0;
    unsigned next_rotate_priority = 0;

    // reset
    loop_rst_IL:
    for (unsigned i = 0; i < num_in_lane; i++) {
        #pragma HLS unroll

        payload_F[i].index = 0;
        payload_A[i].index = 0;
        addr_A[i] = 0;
        payload_C[i].index = 0;
        payload_F[i].data = (PayloadValT){0,0};
        payload_A[i].data = (PayloadValT){0,0};
        payload_C[i].data = (PayloadValT){0,0};
        payload_resend[i].index = 0;
        payload_resend[i].data = (PayloadValT){0,0};

        valid_A[i] = false;
        valid_F[i] = false;
        valid_C[i] = false;

        resend[i] = false;
    }

    loop_reset_OL:
    for (unsigned i = 0; i < num_out_lane; i++) {
        xbar_sel_A[i] = 0;
        xbar_valid_A[i] = false;
        xbar_sel_C[i] = 0;
        xbar_valid_C[i] = false;
    }

    #ifndef __SYNTHESIS__
    int cnt = 0;
    #endif

    loop_shuffle_pipeline:
    while (!loop_exit) {
        #pragma HLS pipeline II=1
        #pragma HLS latency min=7 max=7
        #pragma HLS dependence variable=resend inter distance=6 RAW True
        #pragma HLS dependence variable=payload_resend inter distance=6 RAW True
        #pragma HLS dependence variable=loop_exit inter distance=8 RAW True

        // Fetch stage (F)
        loop_F:
        for (unsigned ILid = 0; ILid < num_in_lane; ILid++) {
            #pragma HLS unroll
            PayloadT payload;
            if (!resend[ILid]) {
                valid_F[ILid] = input_lanes[ILid].read_nb(payload);
            } else {
                payload.data = (PayloadValT){0,0};
                payload.index = 0;
                valid_F[ILid] = true;
            }
            payload_F[ILid] = resend[ILid] ? payload_resend[ILid] : payload;
        }
        // ------- end of F stage

        // Arbiter stage (A)
        loop_A_pass:
        for (unsigned ILid = 0; ILid < num_in_lane; ILid++) {
            #pragma HLS unroll
            // prepare arbiter inputs
            payload_A[ILid] = payload_F[ILid];
            valid_A[ILid] = valid_F[ILid];
            addr_A[ILid] = payload_F[ILid].index;
            arbiter_in_valid[ILid] = valid_F[ILid];
        }
        rotate_priority = next_rotate_priority;
        // pipeline arbiter, depth = 6
        num_granted_A = arbiter_1p<num_in_lane, num_out_lane, addr_mask>(
            addr_A,
            arbiter_in_valid,
            // arbiter_in_granted,
            resend,
            xbar_sel_A,
            xbar_valid_A,
            rotate_priority
        );
        next_rotate_priority = (rotate_priority + 1) % num_in_lane;
        loop_A_fwd:
        for (unsigned ILid = 0; ILid < num_in_lane; ILid++) {
            #pragma HLS unroll
            payload_resend[ILid] = pipereg_stage_A<PayloadT>(payload_A[ILid]);
            // payload_resend[ILid].index = pipereg_stage_A<unsigned>(payload_A[ILid].index);
            // payload_resend[ILid].data = pipereg_stage_A<PayloadValT>(payload_A[ILid].data);
        }
        // ------- end of A stage

        // crossbar stage (C)
        loop_C_pass_il:
        for (unsigned ILid = 0; ILid < num_in_lane; ILid++) {
            #pragma HLS unroll
            payload_C[ILid] = payload_A[ILid];
            valid_C[ILid] = valid_A[ILid];
        }
        num_granted_C = num_granted_A;
        loop_C_pass_ol:
        for (unsigned OLid = 0; OLid < num_in_lane; OLid++) {
            #pragma HLS unroll
            xbar_sel_C[OLid] = xbar_sel_A[OLid];
            xbar_valid_C[OLid] = xbar_valid_A[OLid];
        }
        loop_C_xbar:
        for (unsigned OLid = 0; OLid < num_out_lane; OLid++) {
            #pragma HLS unroll
            if (xbar_valid_C[OLid]) {
                if (valid_C[xbar_sel_C[OLid]]) {
                    output_lanes[OLid].write(payload_C[xbar_sel_C[OLid]]);
                }
            }
        }
        // ------- end of C stage

        if (!prev_finish) { prev_finish = num_payloads_in.read_nb(payload_cnt); }
        process_cnt += num_granted_C;
        bool all_processed = (process_cnt == payload_cnt);
        loop_exit = all_processed && prev_finish;
    }

    num_payloads_out.write(payload_cnt);
}

#endif  // GRAPHLILY_HW_SHUFFLE_H_
