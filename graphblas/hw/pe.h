#ifndef GRAPHLILY_HW_PE_H_
#define GRAPHLILY_HW_PE_H_

#include "hls_stream.h"
#include <iostream>
#include <iomanip>

#include "./util.h"

#ifndef __SYNTHESIS__
bool line_tracing_pe_cluster = false;
#endif

#define MIN(a, b) ((a < b)? a : b)


//----------------------------------------------------------------
// ALUs
//----------------------------------------------------------------

template<typename ValT, typename OpT>
ValT pe_mul_alu(ValT a, ValT b, ValT z, OpT op, bool en) {
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
            out = z;  // TODO: what is z?
            break;
    }
    return en ? out : z;
}


template<typename ValT, typename OpT>
ValT pe_add_alu(ValT a, ValT b, ValT z, OpT op, bool en) {
    #pragma HLS inline
    // #pragma HLS pipeline II=1
    // #pragma HLS latency min=0 max=0
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
// pe_cluster
//----------------------------------------------------------------

// TODO: Do we really need to expose addr_shamt as an argument? Can it be inferred?
template<typename ValT, typename OpT, typename PayloadT,
         unsigned num_PE, unsigned addr_shamt, unsigned bank_size>
void pe_cluster(
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
            incr_M[PEid] = pe_mul_alu<ValT, OpT>(mat_val_M[PEid], vec_val_M[PEid], Zero, Op, valid_M[PEid]);
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
            qt_A[PEid] = pe_add_alu<ValT, OpT>(q0_A[PEid], incr_A[PEid], Zero, Op, valid_A[PEid]);
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

#endif  // GRAPHLILY_HW_PE_H_
