#ifndef GRAPHLILY_HW_UFIXED_PE_H_
#define GRAPHLILY_HW_UFIXED_PE_H_

#include <iostream>
#include <iomanip>

#include "hls_stream.h"

#include "./util.h"


#ifndef __SYNTHESIS__
bool line_tracing_pe_cluster = false;
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
// pe_cluster for bram (read/write latency = 1)
//----------------------------------------------------------------

template<typename ValT, typename OpT, typename PayloadT,
         unsigned num_PE, unsigned addr_shamt, unsigned bank_size>
void ufixed_pe_cluster_spmv_bram(
    hls::stream<PayloadT> input_payloads[num_PE],
    ValT output_buffer[num_PE][bank_size],
    OpT Op,
    ValT Zero,
    hls::stream<unsigned> &num_payloads_in
) {
    // all input fifos are empty
    bool prev_finish = false;
    unsigned npld_dummy;
    bool fifo_allempty = false;
    bool fifo_empty[num_PE];
    #pragma HLS array_partition variable=fifo_empty complete

    // pipeline data registers
    ValT mat_val_F[num_PE];
    ValT vec_val_F[num_PE];
    unsigned row_id_F[num_PE];
    #pragma HLS array_partition variable=mat_val_F complete
    #pragma HLS array_partition variable=vec_val_F complete
    #pragma HLS array_partition variable=row_id_F complete
    ValT mat_val_M[num_PE];
    ValT vec_val_M[num_PE];
    ValT incr_M[num_PE];
    unsigned row_id_M[num_PE];
    #pragma HLS array_partition variable=mat_val_M complete
    #pragma HLS array_partition variable=vec_val_M complete
    #pragma HLS array_partition variable=incr_M complete
    #pragma HLS array_partition variable=row_id_M complete
    ValT incr_R[num_PE];
    ValT q0_R[num_PE];
    unsigned row_id_R[num_PE];
    #pragma HLS array_partition variable=incr_R complete
    #pragma HLS array_partition variable=q0_R complete
    #pragma HLS array_partition variable=row_id_R complete
    ValT incr_A[num_PE];
    ValT q0_A[num_PE];
    ValT qt_A[num_PE];
    unsigned row_id_A[num_PE];
    #pragma HLS array_partition variable=incr_A complete
    #pragma HLS array_partition variable=q0_A complete
    #pragma HLS array_partition variable=qt_A complete
    #pragma HLS array_partition variable=row_id_A complete
    ValT qt_W[num_PE];
    unsigned row_id_W[num_PE];
    #pragma HLS array_partition variable=qt_W complete
    #pragma HLS array_partition variable=row_id_W complete

    // pipeline valid resigters
    bool valid_F[num_PE];
    #pragma HLS array_partition variable=valid_F complete
    bool valid_M[num_PE];
    #pragma HLS array_partition variable=valid_M complete
    bool valid_R[num_PE];
    #pragma HLS array_partition variable=valid_R complete
    bool valid_A[num_PE];
    #pragma HLS array_partition variable=valid_A complete
    bool valid_W[num_PE];
    #pragma HLS array_partition variable=valid_W complete

    // data forwarding flag
    bool fwd_AR[num_PE];
    bool fwd_WR[num_PE];
    #pragma HLS array_partition variable=fwd_AR complete
    #pragma HLS array_partition variable=fwd_WR complete

    // data read from BRAM
    ValT qr[num_PE];
    #pragma HLS array_partition variable=qr complete

    // reset
    loop_rst_PE:
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        mat_val_F[i] = 0;
        vec_val_F[i] = 0;
        row_id_F[i] = 0;
        mat_val_M[i] = 0;
        vec_val_M[i] = 0;
        incr_M[i] = 0;
        row_id_M[i] = 0;
        incr_R[i] = 0;
        q0_R[i] = 0;
        row_id_R[i] = 0;
        incr_A[i] = 0;
        q0_A[i] = 0;
        qt_A[i] = 0;
        row_id_A[i] = 0;
        qt_W[i] = 0;
        row_id_W[i] = 0;

        valid_F[i] = false;
        valid_M[i] = false;
        valid_R[i] = false;
        valid_A[i] = false;
        valid_W[i] = false;

        fwd_AR[i] = false;
        fwd_WR[i] = false;
        qr[i] = 0;
    }

    #ifndef __SYNTHESIS__
    int cnt = 0;
    #endif

    loop_process_pipeline:
    while (!(prev_finish && fifo_allempty)) {
        #pragma HLS pipeline II=1
        #pragma HLS dependence variable=output_buffer inter distance=2 RAW True

        // #pragma HLS dependence variable=valid_A  inter distance=1 RAW True
        // #pragma HLS dependence variable=row_id_A inter distance=1 RAW True
        // #pragma HLS dependence variable=qt_A     inter distance=1 RAW True
        // #pragma HLS dependence variable=valid_W  inter distance=2 RAW True
        // #pragma HLS dependence variable=row_id_W inter distance=2 RAW True
        // #pragma HLS dependence variable=qt_W     inter distance=2 RAW True

        if (!prev_finish) { prev_finish = num_payloads_in.read_nb(npld_dummy); }

        // Fetch stage (F)
        loop_F:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            PayloadT payload;
            fifo_empty[PEid] = !input_payloads[PEid].read_nb(payload);
            if (fifo_empty[PEid]) {
                mat_val_F[PEid] = 0;
                vec_val_F[PEid] = 0;
                row_id_F[PEid] = 0;
                valid_F[PEid] = false;
            } else {
                mat_val_F[PEid] = payload.data.mat_val;
                vec_val_F[PEid] = payload.data.vec_val;
                row_id_F[PEid] = payload.index;
                valid_F[PEid] = true;
            }
        }
        fifo_allempty = array_and_reduction<num_PE>(fifo_empty);
        // ------- end of F stage

        // Mul stage (M)
        loop_M:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            valid_M[PEid] = valid_F[PEid];
            row_id_M[PEid] = row_id_F[PEid];
            mat_val_M[PEid] = mat_val_F[PEid];
            vec_val_M[PEid] = vec_val_F[PEid];
            incr_M[PEid] = pe_ufixed_mul_alu<ValT, OpT>(mat_val_M[PEid], vec_val_M[PEid], Zero, Op, valid_M[PEid]);
        }
        // ------- end of M stage

        // Read stage (R)
        loop_R:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            valid_R[PEid] = valid_M[PEid];
            row_id_R[PEid] = row_id_M[PEid];
            incr_R[PEid] = incr_M[PEid];
            if (valid_R[PEid]) {
                qr[PEid] = output_buffer[PEid][row_id_R[PEid] >> addr_shamt];
            } else {
                qr[PEid] = 0;
            }
            // data forwarding control
            fwd_AR[PEid] = valid_R[PEid] &&
                           valid_A[PEid] &&
                           ((row_id_R[PEid] >> addr_shamt) == (row_id_A[PEid] >> addr_shamt));
            fwd_WR[PEid] = valid_R[PEid] &&
                           valid_W[PEid] &&
                           ((row_id_R[PEid] >> addr_shamt) == (row_id_W[PEid] >> addr_shamt));
            q0_R[PEid] = fwd_AR[PEid] ? qt_A[PEid] : (fwd_WR[PEid] ? qt_W[PEid] : qr[PEid]);
        }
        // ------- end of R stage

        // Accumulate stage (A)
        loop_A:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            valid_A[PEid] = valid_R[PEid];
            row_id_A[PEid] = row_id_R[PEid];
            q0_A[PEid] = q0_R[PEid];
            incr_A[PEid] = incr_R[PEid];
            qt_A[PEid] = pe_ufixed_add_alu<ValT, OpT>(q0_A[PEid], incr_A[PEid], Zero, Op, valid_A[PEid]);
        }
        // ------- end of A stage

        // Write stage (W)
        loop_W:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            valid_W[PEid] = valid_A[PEid];
            row_id_W[PEid] = row_id_A[PEid];
            qt_W[PEid] = qt_A[PEid];
            if (valid_W[PEid]) {
                output_buffer[PEid][row_id_W[PEid] >> addr_shamt] = qt_W[PEid];
            }
        }
        // ------- end of W stage

        // line tracing
        #ifndef __SYNTHESIS__
        if (line_tracing_pe_cluster) {
            auto f2s = [](float value) {
                char s[9];
                sprintf(s, "%4.3f", value);
                std::string str = s;
                return str;
            };
            // if (cnt > 10) {
            //     std::cout << "ERROR: [pe-csim] simulation exceed max loop count limit!" << std::endl;
            //     break;
            // }
            cnt++;
            std::cout << std::setw(3) << "loop cnt: " << cnt << " {"
                      << std::setw(2) << (prev_finish   ? "PF" : "..") << "|"
                      << std::setw(2) << (fifo_allempty ? "AF" : "..") << "" << "}"
                      << std::endl << std::flush;
            for (size_t x = 0; x < num_PE; x++) {
                std::cout << "  PE[" << std::setw(2) << x << "]{";
                std::cout << (valid_F[x] ? "F" : ".") << "|";
                std::cout << (valid_M[x] ? "M" : ".") << " <"
                          << std::setw(8) << (valid_M[x] ? f2s((float)mat_val_F[x])   : "--") << " * "
                          << std::setw(8) << (valid_M[x] ? f2s((float)vec_val_F[x])   : "--") << " = "
                          << std::setw(8) << (valid_M[x] ? f2s((float)incr_M[x])      : "--") << ", "
                          << std::setw(8) << (valid_M[x] ? std::to_string(row_id_M[x]): "--") << "> " << "|";
                std::cout << (valid_R[x] ? "R" : ".") << " <"
                          << std::setw(8) << (valid_R[x] ? std::to_string(row_id_R[x]): "--") << " ~ "
                          << std::setw(8) << (valid_R[x] ? f2s((float)q0_R[x])        : "--") << ", "
                          << std::setw(2) << (fwd_AR[x]  ? "fA"                       : "--") << ", "
                          << std::setw(2) << (fwd_WR[x]  ? "fW"                       : "--") << "> " << "|";
                std::cout << (valid_A[x] ? "A" : ".") << " <"
                          << std::setw(8) << (valid_A[x] ? f2s((float)incr_R[x])      : "--") << " + "
                          << std::setw(8) << (valid_A[x] ? f2s((float)q0_R[x])        : "--") << " = "
                          << std::setw(8) << (valid_A[x] ? f2s((float)qt_A[x])        : "--") << " ~ "
                          << std::setw(8) << (valid_A[x] ? std::to_string(row_id_A[x]): "--") << "> " << "|";
                std::cout << (valid_W[x] ? "W" : ". ");
                std::cout << std::endl << std::flush;
            }
        }
        #endif
    }
}


template<typename ValT, typename OpT, typename PayloadT,
         unsigned num_hbm_channels, unsigned num_PE, unsigned addr_shamt, unsigned bank_size>
void ufixed_pe_cluster_spmspv_bram(
    hls::stream<PayloadT> input_payloads[num_PE],
    ValT output_buffer[num_hbm_channels][num_PE][bank_size],
    OpT Op,
    ValT Zero,
    hls::stream<unsigned> &num_payloads_in
) {
    // all input fifos are empty
    bool prev_finish = false;
    unsigned npld_dummy;
    bool fifo_allempty = false;
    bool fifo_empty[num_PE];
    #pragma HLS array_partition variable=fifo_empty complete

    // pipeline data registers
    ValT mat_val_F[num_PE];
    ValT vec_val_F[num_PE];
    unsigned row_id_F[num_PE];
    #pragma HLS array_partition variable=mat_val_F complete
    #pragma HLS array_partition variable=vec_val_F complete
    #pragma HLS array_partition variable=row_id_F complete
    ValT mat_val_M[num_PE];
    ValT vec_val_M[num_PE];
    ValT incr_M[num_PE];
    unsigned row_id_M[num_PE];
    #pragma HLS array_partition variable=mat_val_M complete
    #pragma HLS array_partition variable=vec_val_M complete
    #pragma HLS array_partition variable=incr_M complete
    #pragma HLS array_partition variable=row_id_M complete
    ValT incr_R[num_PE];
    ValT q0_R[num_PE];
    unsigned row_id_R[num_PE];
    #pragma HLS array_partition variable=incr_R complete
    #pragma HLS array_partition variable=q0_R complete
    #pragma HLS array_partition variable=row_id_R complete
    ValT incr_A[num_PE];
    ValT q0_A[num_PE];
    ValT qt_A[num_PE];
    unsigned row_id_A[num_PE];
    #pragma HLS array_partition variable=incr_A complete
    #pragma HLS array_partition variable=q0_A complete
    #pragma HLS array_partition variable=qt_A complete
    #pragma HLS array_partition variable=row_id_A complete
    ValT qt_W[num_PE];
    unsigned row_id_W[num_PE];
    #pragma HLS array_partition variable=qt_W complete
    #pragma HLS array_partition variable=row_id_W complete

    // pipeline valid resigters
    bool valid_F[num_PE];
    #pragma HLS array_partition variable=valid_F complete
    bool valid_M[num_PE];
    #pragma HLS array_partition variable=valid_M complete
    bool valid_R[num_PE];
    #pragma HLS array_partition variable=valid_R complete
    bool valid_A[num_PE];
    #pragma HLS array_partition variable=valid_A complete
    bool valid_W[num_PE];
    #pragma HLS array_partition variable=valid_W complete

    // data forwarding flag
    bool fwd_AR[num_PE];
    bool fwd_WR[num_PE];
    #pragma HLS array_partition variable=fwd_AR complete
    #pragma HLS array_partition variable=fwd_WR complete

    // data read from BRAM
    ValT qr[num_PE];
    #pragma HLS array_partition variable=qr complete

    // reset
    loop_rst_PE:
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        mat_val_F[i] = 0;
        vec_val_F[i] = 0;
        row_id_F[i] = 0;
        mat_val_M[i] = 0;
        vec_val_M[i] = 0;
        incr_M[i] = 0;
        row_id_M[i] = 0;
        incr_R[i] = 0;
        q0_R[i] = 0;
        row_id_R[i] = 0;
        incr_A[i] = 0;
        q0_A[i] = 0;
        qt_A[i] = 0;
        row_id_A[i] = 0;
        qt_W[i] = 0;
        row_id_W[i] = 0;

        valid_F[i] = false;
        valid_M[i] = false;
        valid_R[i] = false;
        valid_A[i] = false;
        valid_W[i] = false;

        fwd_AR[i] = false;
        fwd_WR[i] = false;
        qr[i] = 0;
    }

    #ifndef __SYNTHESIS__
    int cnt = 0;
    #endif

    loop_process_pipeline:
    while (!(prev_finish && fifo_allempty)) {
        #pragma HLS pipeline II=1
        #pragma HLS dependence variable=output_buffer inter distance=2 RAW True

        // #pragma HLS dependence variable=valid_A  inter distance=1 RAW True
        // #pragma HLS dependence variable=row_id_A inter distance=1 RAW True
        // #pragma HLS dependence variable=qt_A     inter distance=1 RAW True
        // #pragma HLS dependence variable=valid_W  inter distance=2 RAW True
        // #pragma HLS dependence variable=row_id_W inter distance=2 RAW True
        // #pragma HLS dependence variable=qt_W     inter distance=2 RAW True

        if (!prev_finish) { prev_finish = num_payloads_in.read_nb(npld_dummy); }

        // Fetch stage (F)
        loop_F:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            PayloadT payload;
            fifo_empty[PEid] = !input_payloads[PEid].read_nb(payload);
            if (fifo_empty[PEid]) {
                mat_val_F[PEid] = 0;
                vec_val_F[PEid] = 0;
                row_id_F[PEid] = 0;
                valid_F[PEid] = false;
            } else {
                mat_val_F[PEid] = payload.data.mat_val;
                vec_val_F[PEid] = payload.data.vec_val;
                row_id_F[PEid] = payload.index;
                valid_F[PEid] = true;
            }
        }
        fifo_allempty = array_and_reduction<num_PE>(fifo_empty);
        // ------- end of F stage

        // Mul stage (M)
        loop_M:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            valid_M[PEid] = valid_F[PEid];
            row_id_M[PEid] = row_id_F[PEid];
            mat_val_M[PEid] = mat_val_F[PEid];
            vec_val_M[PEid] = vec_val_F[PEid];
            incr_M[PEid] = pe_ufixed_mul_alu<ValT, OpT>(mat_val_M[PEid], vec_val_M[PEid], Zero, Op, valid_M[PEid]);
        }
        // ------- end of M stage

        // Read stage (R)
        loop_R:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            valid_R[PEid] = valid_M[PEid];
            row_id_R[PEid] = row_id_M[PEid];
            incr_R[PEid] = incr_M[PEid];
            if (valid_R[PEid]) {
                qr[PEid] = output_buffer[(row_id_R[PEid] >> addr_shamt) % num_hbm_channels][PEid]
                                        [(row_id_R[PEid] >> addr_shamt) / num_hbm_channels];
            } else {
                qr[PEid] = 0;
            }
            // data forwarding control
            fwd_AR[PEid] = valid_R[PEid] &&
                           valid_A[PEid] &&
                           ((row_id_R[PEid] >> addr_shamt) == (row_id_A[PEid] >> addr_shamt));
            fwd_WR[PEid] = valid_R[PEid] &&
                           valid_W[PEid] &&
                           ((row_id_R[PEid] >> addr_shamt) == (row_id_W[PEid] >> addr_shamt));
            q0_R[PEid] = fwd_AR[PEid] ? qt_A[PEid] : (fwd_WR[PEid] ? qt_W[PEid] : qr[PEid]);
        }
        // ------- end of R stage

        // Accumulate stage (A)
        loop_A:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            valid_A[PEid] = valid_R[PEid];
            row_id_A[PEid] = row_id_R[PEid];
            q0_A[PEid] = q0_R[PEid];
            incr_A[PEid] = incr_R[PEid];
            qt_A[PEid] = pe_ufixed_add_alu<ValT, OpT>(q0_A[PEid], incr_A[PEid], Zero, Op, valid_A[PEid]);
        }
        // ------- end of A stage

        // Write stage (W)
        loop_W:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            valid_W[PEid] = valid_A[PEid];
            row_id_W[PEid] = row_id_A[PEid];
            qt_W[PEid] = qt_A[PEid];
            if (valid_W[PEid]) {
                output_buffer[(row_id_R[PEid] >> addr_shamt) % num_hbm_channels][PEid]
                             [(row_id_R[PEid] >> addr_shamt) / num_hbm_channels] = qt_W[PEid];
            }
        }
        // ------- end of W stage
    }
}


//----------------------------------------------------------------
// pe_cluster for uram (read/write latency = 2)
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
*/
template<typename ValT>
struct PP_STREAM_T {
    unsigned addr;
    ValT incr;
};


template<typename ValT, typename OpT, typename PayloadT, unsigned num_PE>
void ufixed_pe_cluster_uram_part1(
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
    }
    num_payloads_out.write(num_payload);
}


template<typename ValT, typename OpT, unsigned num_PE, unsigned bank_size, unsigned addr_shamt>
void ufixed_pe_cluster_spmv_uram_part2(
    hls::stream<PP_STREAM_T<ValT> > input_payloads[num_PE],
    hls::stream<unsigned> &num_payloads_in,
    ValT output_buffer[num_PE][bank_size],
    OpT Op,
    ValT Zero
) {
    // loop control
    bool prev_finish = false;
    unsigned num_payload = 0;
    unsigned process_cnt = 0;
    bool loop_exit = false;

    // S stage (depth 1)
    unsigned addr_S[num_PE];
    #pragma HLS array_partition variable=addr_S complete
    ValT incr_S[num_PE];
    #pragma HLS array_partition variable=incr_S complete
    bool valid_S[num_PE];
    #pragma HLS array_partition variable=valid_S complete
    bool next_valid_R[num_PE];
    #pragma HLS array_partition variable=next_valid_R complete
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        addr_S[i] = 0;
        incr_S[i] = 0;
        valid_S[i] = false;
        next_valid_R[i] = false;
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
    unsigned in_flight_R1_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_R1_addr complete
    bool in_flight_R1_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_R1_valid complete
    unsigned in_flight_R2_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_R1_addr complete
    bool in_flight_R2_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_R1_valid complete
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        q_R[i] = 0;
        incr_R[i] = 0;
        addr_R[i] = 0;
        valid_R[i] = false;
        in_flight_R1_addr[i] = 0;
        in_flight_R2_addr[i] = 0;
        in_flight_R1_valid[i] = false;
        in_flight_R2_valid[i] = false;
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
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        match_SR[i] = false;
        match_SA[i] = false;
        match_SW[i] = false;
        stall_S[i] = false;
    }

    loop_pe_part2:
    while (!loop_exit) {
        #pragma HLS pipeline II=1
        #pragma HLS latency min=6 max=6
        #pragma HLS dependence variable=output_buffer inter RAW false
        #pragma HLS dependence variable=loop_exit inter RAW true distance=7

        // Stall stage (S)
        loop_S:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            PP_STREAM_T<ValT> payload_in_tmp;
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

            match_SR[PEid] = ((addr_S[PEid] == in_flight_R1_addr[PEid]) && valid_S[PEid] && in_flight_R1_valid[PEid])
                          || ((addr_S[PEid] == in_flight_R2_addr[PEid]) && valid_S[PEid] && in_flight_R2_valid[PEid]);
            match_SA[PEid] = ((addr_S[PEid] == in_flight_A1_addr[PEid]) && valid_S[PEid] && in_flight_A1_valid[PEid]);
            match_SW[PEid] = ((addr_S[PEid] == in_flight_W1_addr[PEid]) && valid_S[PEid] && in_flight_W1_valid[PEid])
                          || ((addr_S[PEid] == in_flight_W2_addr[PEid]) && valid_S[PEid] && in_flight_W2_valid[PEid]);
            stall_S[PEid] = match_SR[PEid] || match_SA[PEid] || match_SW[PEid];
            next_valid_R[PEid] = valid_S[PEid] && !stall_S[PEid];
        }
        // ----- end of S stage

        // update in-flight registers
        loop_update_in_flight:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            in_flight_W2_valid[PEid] = in_flight_W1_valid[PEid];
            in_flight_W1_valid[PEid] = in_flight_A1_valid[PEid];
            in_flight_A1_valid[PEid] = in_flight_R1_valid[PEid];
            in_flight_R2_valid[PEid] = in_flight_R1_valid[PEid];
            in_flight_R1_valid[PEid] = next_valid_R[PEid];

            in_flight_W2_addr[PEid] = in_flight_W1_addr[PEid];
            in_flight_W1_addr[PEid] = in_flight_A1_addr[PEid];
            in_flight_A1_addr[PEid] = in_flight_R1_addr[PEid];
            in_flight_R2_addr[PEid] = in_flight_R1_addr[PEid];
            in_flight_R1_addr[PEid] = addr_S[PEid];
        }

        // Read stage (R)
        loop_R:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
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
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            addr_A[PEid] = addr_R[PEid];
            valid_A[PEid] = valid_R[PEid];
            q_A[PEid] = q_R[PEid];
            incr_A[PEid] = incr_R[PEid];
            new_q_A[PEid] = pe_ufixed_add_alu<ValT, OpT>(q_A[PEid], incr_A[PEid], Zero, Op, valid_A[PEid]);
        }
        // ----- end of A stage

        // Write stage (W)
        unsigned process_cnt_incr = 0;
        loop_W:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            addr_W[PEid] = addr_A[PEid];
            valid_W[PEid] = valid_A[PEid];
            new_q_W[PEid] = new_q_A[PEid];
            if (valid_W[PEid]) {
                output_buffer[PEid][addr_W[PEid] >> addr_shamt] = new_q_W[PEid];
                process_cnt_incr++;
            }
        }
        if (!prev_finish) {
            prev_finish = num_payloads_in.read_nb(num_payload);
        }
        process_cnt += process_cnt_incr;
        bool process_complete = (process_cnt == num_payload);
        loop_exit = process_complete && prev_finish;
    }
}


template<typename ValT, typename OpT, typename PayloadT,
         unsigned num_PE, unsigned addr_shamt, unsigned bank_size>
void ufixed_pe_cluster_spmv_uram(
    hls::stream<PayloadT> input_payloads[num_PE],
    ValT output_buffer[num_PE][bank_size],
    OpT Op,
    ValT Zero,
    hls::stream<unsigned> &num_payloads_in
) {
    #pragma HLS dataflow

    hls::stream<PP_STREAM_T<ValT> > PP_stream[num_PE];
    #pragma HLS stream variable=PP_stream depth=4
    hls::stream<unsigned> PP_npld_stream;
    #pragma HLS stream variable=PP_npld_stream depth=2

    ufixed_pe_cluster_uram_part1<ValT, OpT, PayloadT, num_PE>(input_payloads, PP_stream,
        num_payloads_in, PP_npld_stream, Op, Zero);
    ufixed_pe_cluster_spmv_uram_part2<ValT, OpT, num_PE, bank_size, addr_shamt>(PP_stream, PP_npld_stream,
        output_buffer, Op, Zero);
}


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

    // S stage (depth 1)
    unsigned addr_S[num_PE];
    #pragma HLS array_partition variable=addr_S complete
    ValT incr_S[num_PE];
    #pragma HLS array_partition variable=incr_S complete
    bool valid_S[num_PE];
    #pragma HLS array_partition variable=valid_S complete
    bool next_valid_R[num_PE];
    #pragma HLS array_partition variable=next_valid_R complete
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        addr_S[i] = 0;
        incr_S[i] = 0;
        valid_S[i] = false;
        next_valid_R[i] = false;
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
    unsigned in_flight_R1_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_R1_addr complete
    bool in_flight_R1_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_R1_valid complete
    unsigned in_flight_R2_addr[num_PE];
    #pragma HLS array_partition variable=in_flight_R1_addr complete
    bool in_flight_R2_valid[num_PE];
    #pragma HLS array_partition variable=in_flight_R1_valid complete
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        q_R[i] = 0;
        incr_R[i] = 0;
        addr_R[i] = 0;
        valid_R[i] = false;
        in_flight_R1_addr[i] = 0;
        in_flight_R2_addr[i] = 0;
        in_flight_R1_valid[i] = false;
        in_flight_R2_valid[i] = false;
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
    for (unsigned i = 0; i < num_PE; i++) {
        #pragma HLS unroll
        match_SR[i] = false;
        match_SA[i] = false;
        match_SW[i] = false;
        stall_S[i] = false;
    }

    loop_pe_part2:
    while (!loop_exit) {
        #pragma HLS pipeline II=1
        #pragma HLS latency min=6 max=6
        #pragma HLS dependence variable=output_buffer inter RAW false
        #pragma HLS dependence variable=loop_exit inter RAW true distance=7

        // Skid stage (S)
        loop_S:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            PP_STREAM_T<ValT> payload_in_tmp;
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

            match_SR[PEid] = ((addr_S[PEid] == in_flight_R1_addr[PEid]) && valid_S[PEid] && in_flight_R1_valid[PEid])
                          || ((addr_S[PEid] == in_flight_R2_addr[PEid]) && valid_S[PEid] && in_flight_R2_valid[PEid]);
            match_SA[PEid] = ((addr_S[PEid] == in_flight_A1_addr[PEid]) && valid_S[PEid] && in_flight_A1_valid[PEid]);
            match_SW[PEid] = ((addr_S[PEid] == in_flight_W1_addr[PEid]) && valid_S[PEid] && in_flight_W1_valid[PEid])
                          || ((addr_S[PEid] == in_flight_W2_addr[PEid]) && valid_S[PEid] && in_flight_W2_valid[PEid]);
            stall_S[PEid] = match_SR[PEid] || match_SA[PEid] || match_SW[PEid];
            next_valid_R[PEid] = valid_S[PEid] && !stall_S[PEid];
        }
        // ----- end of S stage

        // update in-flight registers
        loop_update_in_flight:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            in_flight_W2_valid[PEid] = in_flight_W1_valid[PEid];
            in_flight_W1_valid[PEid] = in_flight_A1_valid[PEid];
            in_flight_A1_valid[PEid] = in_flight_R1_valid[PEid];
            in_flight_R2_valid[PEid] = in_flight_R1_valid[PEid];
            in_flight_R1_valid[PEid] = next_valid_R[PEid];

            in_flight_W2_addr[PEid] = in_flight_W1_addr[PEid];
            in_flight_W1_addr[PEid] = in_flight_A1_addr[PEid];
            in_flight_A1_addr[PEid] = in_flight_R1_addr[PEid];
            in_flight_R2_addr[PEid] = in_flight_R1_addr[PEid];
            in_flight_R1_addr[PEid] = addr_S[PEid];
        }

        // Read stage (R)
        loop_R:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            valid_R[PEid] = next_valid_R[PEid];
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

        // Add stage (A)
        loop_A:
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            #pragma HLS unroll
            addr_A[PEid] = addr_R[PEid];
            valid_A[PEid] = valid_R[PEid];
            q_A[PEid] = q_R[PEid];
            incr_A[PEid] = incr_R[PEid];
            new_q_A[PEid] = pe_ufixed_add_alu<ValT, OpT>(q_A[PEid], incr_A[PEid], Zero, Op, valid_A[PEid]);
        }
        // ----- end of A stage

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
    }
}


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

    ufixed_pe_cluster_uram_part1<ValT, OpT, PayloadT, num_PE>(input_payloads,
        PP_stream, num_payloads_in, PP_npld_stream, Op, Zero);
    ufixed_pe_cluster_spmspv_uram_part2<ValT, OpT, num_hbm_channels, num_PE, bank_size, addr_shamt>(PP_stream,
        PP_npld_stream, output_buffer, Op, Zero);
}

#endif  // GRAPHLILY_HW_UFIXED_PE_H_
