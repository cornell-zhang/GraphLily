#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <assert.h>
#include <iostream>
#include <iomanip>

#include "./kernel_spmspv.h"


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

//----------------------------------------------------
// Data Loader
//----------------------------------------------------

static void DL_spmspv(
    // bram buffers
    const PACKED_DWI_T mat_dwi[],
    const INDEX_T mat_idxptr[],
    const DIT_T vec_dit[],
    // stream buffers (DL->PE)
    hls::stream<VAL_T> nnz_from_mat_stream[],
    hls::stream<INDEX_T> current_row_id_stream[],
    hls::stream<VAL_T> &nnz_from_vec_stream,
    hls::stream<INDEX_T> &current_collen_stream,
    // column count
    unsigned int vec_nnz_total,
    // mat size
    const unsigned int num_columns,
    // tile count
    unsigned int tile_cnt,
    // tile base
    INDEX_T tile_base
) {
    // calculate mat_idxptr base address
    unsigned int idxptr_base = tile_cnt * (num_columns + 1);

    // line tracing
    #ifndef __SYNTHESIS__
        std::cout << "DL idxptr base: " << std::setw(5) << idxptr_base << std::endl;
    #endif

    // loop over all active columns
    loop_over_active_columns_DL:
    for (unsigned int vec_nnz_cnt = 0; vec_nnz_cnt < vec_nnz_total; vec_nnz_cnt++) {

        // slice out the current column out of the active columns
        INDEX_T current_colid = vec_dit[vec_nnz_cnt + 1].index;
        nnz_from_vec_stream << vec_dit[vec_nnz_cnt + 1].data;

        // [0] for start, [1] for end
        INDEX_T col_slice[2];
        #pragma HLS array_partition variable=col_slice complete

        // line tracing
        // #ifndef __SYNTHESIS__
        //     std::cout << "DL Reading Idxptr from : "
        //               << "Start[" << std::setw(5) << current_colid + idxptr_base     << "], "
        //               << "End  [" << std::setw(5) << current_colid + idxptr_base + 1 << "]" << std::endl;
        // #endif

        loop_get_column_len_DL_unroll:
        for (unsigned int i = 0; i < 2; i++) {
            #pragma HLS unroll
            col_slice[i] = mat_idxptr[current_colid + idxptr_base + i];
        }
        INDEX_T current_collen = col_slice[1] - col_slice[0];

        // line tracing
        // #ifndef __SYNTHESIS__
        //     std::cout << "DL Active Column : "
        //               << "Start" << std::setw(5) << col_slice[0]     << ", "
        //               << "End  " << std::setw(5) << col_slice[1] - 1 << ", "
        //               << "Size " << std::setw(3) << current_collen   << std::endl;
        // #endif

        current_collen_stream << current_collen; // measured in number of packets

        loop_over_pkts_DL:
        for (unsigned int i = 0; i < current_collen; i++) {
            #pragma HLS pipeline II=1

            // line tracing
            // #ifndef __SYNTHESIS__
            //     std::cout << "DL Loading from: "
            //               << "Pkt[" << std::setw(5) << tile_base + i + col_slice[0] << "]" << std::endl;
            // #endif

            // [IMPORTANT] read mat_dit here
            PACKED_DWI_T dwi_packet_from_mat = mat_dwi[tile_base + i + col_slice[0]];

            loop_unpack_DL_unroll:
            for (unsigned int k = 0; k < PACKET_SIZE; k++) {
                #pragma HLS unroll
                nnz_from_mat_stream[k] << dwi_packet_from_mat.datapkt[k];
                current_row_id_stream[k] << dwi_packet_from_mat.indexpkt[k];
            }
        }
    }
}

//------------------------------------------------------------
// array shift functions
//------------------------------------------------------------

template <typename T, const unsigned int ARRAY_SIZE>
void array_shift_left(T array[ARRAY_SIZE],T array_dest[ARRAY_SIZE], unsigned int rotate) {
    #pragma HLS inline
    // #pragma HLS latency min=0 max=0
    #pragma HLS array_partition variable=array complete
    #pragma HLS array_partition variable=array_dest complete
    for (int i = 0; i < ARRAY_SIZE; i++) {
        #pragma HLS UNROLL
        array_dest[i] = array[(i + rotate) % ARRAY_SIZE];
    }
}

//------------------------------------------------------------
// array add functions
//------------------------------------------------------------

template <typename INDEX_T, const unsigned int ARRAY_SIZE, const unsigned int MAX>
void array_cyclic_add(arbiter_result<INDEX_T> array[ARRAY_SIZE], unsigned int inc) {
    #pragma HLS inline
    // #pragma HLS latency min=0 max=0
    #pragma HLS array_partition variable=array complete
    for (unsigned int i = 0; i < ARRAY_SIZE; i++) {
        #pragma HLS unroll
        if(!array[i].bank_idle) {
            array[i].virtual_port_id = (array[i].virtual_port_id + inc) % MAX;
        }
    }
}

//------------------------------------------------------------
// bool array reduction functions
//------------------------------------------------------------

// reduction and
template <const unsigned int ARRAY_SIZE>
bool bool_array_and_reduction(bool array1[ARRAY_SIZE],bool array2[ARRAY_SIZE]) {
    #pragma HLS latency min=0 max=0
    bool result = true;
    for (unsigned int i = 0; i < ARRAY_SIZE; i++) {
        #pragma HLS unroll
        result = result && (array1[i] && array2[i]);
    }
    return result;
}

// reduction or
template <const unsigned int ARRAY_SIZE>
bool bool_array_or_reduction(bool array[ARRAY_SIZE]) {
    #pragma HLS latency min=0 max=0
    bool result = false;
    for (unsigned int i = 0; i < ARRAY_SIZE; i++) {
        #pragma HLS unroll
        result = result || array[i];
    }
    return result;
}


//------------------------------------------------------------
// bram access network (BAN) functions
//------------------------------------------------------------

// arbiter logic
void bram_access_arbiter(
    // virtual ports
    rd_req<INDEX_T> requests[NUM_PE],
    // arbitering results (used for write)
    arbiter_result<INDEX_T> rd_arbiter_results[NUM_BANK],
    // grant PE (used to grant PEs)
    bool granted[NUM_PE],
    // rotate priority
    unsigned int rotate_priority
){
    #pragma HLS latency min=ARBITER_LATENCY max=ARBITER_LATENCY
    #pragma HLS pipeline II=1

    rd_req<INDEX_T> requests_temp[NUM_PE];
    #pragma HLS array_partition variable=requests_temp complete

    bool good_req[NUM_PE];
    #pragma HLS array_partition variable=good_req complete

    // change to the correct priority
    array_shift_left<rd_req<INDEX_T>,NUM_PE>(requests,requests_temp,rotate_priority);

    // find out requests to be arbitered
    loop_ab_check_req_unroll:
    for (unsigned int VPid = 0; VPid < NUM_PE; VPid++) {
        #pragma HLS unroll
        good_req[VPid] = (requests_temp[VPid].valid) && (!requests_temp[VPid].zero);
    }

    // arbiter

    loop_ab_logic_unroll:
    for(unsigned int BKid = 0; BKid < NUM_BANK; BKid++) {
        #pragma HLS unroll
        bool found = false;
        INDEX_T chosen_port = 0;

        loop_ab_logic_encoder_unroll:
        for(unsigned int VPid_plus_1 = NUM_PE; VPid_plus_1 > 0; VPid_plus_1 --) {
            #pragma HLS unroll
            if(good_req[VPid_plus_1 - 1] && ((requests_temp[VPid_plus_1 - 1].addr & BANK_ID_MASK) == BKid)) {
                chosen_port = VPid_plus_1 - 1;
                found = true;
            }
        }

        if(!found) {
            rd_arbiter_results[BKid].bank_idle = true;
            rd_arbiter_results[BKid].virtual_port_id = 0;
        } else {
            rd_arbiter_results[BKid].bank_idle = false;
            rd_arbiter_results[BKid].virtual_port_id = chosen_port;
        }

    }

    // adjust airbiter results back
    array_cyclic_add<INDEX_T,NUM_BANK,NUM_PE>(rd_arbiter_results,rotate_priority);

    // grant PEs
    loop_grant_PE_unroll:
    for (unsigned int VPid = 0; VPid < NUM_PE; VPid++) {
        #pragma HLS unroll
        // count zero requests and granted requests
        INDEX_T bkid = requests[VPid].addr & BANK_ID_MASK;
        granted[VPid] = (!rd_arbiter_results[bkid].bank_idle) && (rd_arbiter_results[bkid].virtual_port_id == VPid);
    }
}


void bram_access_read(
    // virtual ports
    rd_req<INDEX_T> requests[NUM_PE],
    rd_resp<VAL_T> responses[NUM_PE],
    // bram
    VAL_T bram[NUM_BANK][BANK_SIZE],
    // arbitering results (used for write)
    arbiter_result<INDEX_T> rd_arbiter_results[NUM_BANK],
    bool granted_PE[NUM_PE]
){
    #pragma HLS inline

    // pipeline registers (_local means local to bram)
    rd_req<INDEX_T> requests_local[NUM_PE];
    rd_resp<VAL_T> responses_local[NUM_PE];
    arbiter_result<INDEX_T> rd_arbiter_results_local[NUM_BANK];
    bool granted_PE_local[NUM_PE];
    #pragma HLS array_partition variable=requests_local complete
    #pragma HLS array_partition variable=response_local complete
    #pragma HLS array_partition variable=rd_arbiter_results_local complete
    #pragma HLS array_partition variable=granted_PE_local complete

    // input pipeline
    loop_rb_inpp_abresult_unroll:
    for (unsigned int BKid = 0; BKid < NUM_BANK; BKid++) {
        #pragma HLS unroll
        rd_arbiter_results_local[BKid] =
            HLS_REG< arbiter_result<INDEX_T> >(
                HLS_REG< arbiter_result<INDEX_T> >(
                    HLS_REG< arbiter_result<INDEX_T> >(
                        HLS_REG< arbiter_result<INDEX_T> >(
                            HLS_REG< arbiter_result<INDEX_T> >(rd_arbiter_results[BKid])))));
    }
    loop_rb_inpp_vpreq_unroll:
    for (unsigned int VPid = 0; VPid < NUM_PE; VPid++) {
        #pragma HLS unroll
        requests_local[VPid] =
            HLS_REG< rd_req<INDEX_T> >(
                HLS_REG< rd_req<INDEX_T> >(
                    HLS_REG< rd_req<INDEX_T> >(
                        HLS_REG< rd_req<INDEX_T> >(
                            HLS_REG< rd_req<INDEX_T> >(requests[VPid])))));
        granted_PE_local[VPid] =
            HLS_REG< bool >(
                HLS_REG< bool >(
                    HLS_REG< bool >(
                        HLS_REG< bool >(
                            HLS_REG< bool >(granted_PE[VPid])))));
    }

    // read access logic
    VAL_T rd_data_local[NUM_BANK];
    #pragma HLS array_partition variable=rd_data_local complete

    // first get read data
    loop_rd_get_data_unroll:
    for(unsigned int BKid = 0; BKid < NUM_BANK; BKid++) {
        #pragma HLS unroll
        if(!rd_arbiter_results_local[BKid].bank_idle) {
            INDEX_T vpid = rd_arbiter_results_local[BKid].virtual_port_id;
            rd_data_local[BKid] = bram[BKid][requests_local[vpid].addr >> BANK_ID_NBITS];
        } else {
            rd_data_local[BKid] = 0;
        }
    }

    // then send data back to vritual ports
    loop_rd_port_resp_unroll:
    for (unsigned int VPid = 0; VPid < NUM_PE; VPid++) {
        #pragma HLS unroll
        if(granted_PE_local[VPid]){
            responses_local[VPid].data = rd_data_local[requests_local[VPid].addr & BANK_ID_MASK];
            responses_local[VPid].valid = 1;
        } else {
            responses_local[VPid].data = 0;
            responses_local[VPid].valid = 0;
        }
    }

    // output pipeline
    loop_rb_outpp_vpresp_unroll:
    for(unsigned int VPid = 0; VPid < NUM_PE; VPid++){
        #pragma HLS unroll
        responses[VPid] =
            HLS_REG< rd_resp<VAL_T> >(
                HLS_REG< rd_resp<VAL_T> >(
                    HLS_REG< rd_resp<VAL_T> >(
                        HLS_REG< rd_resp<VAL_T> >(
                            HLS_REG< rd_resp<VAL_T> >(responses_local[VPid])))));
    }
}

// write logic
void bram_access_write(
    // virtual ports
    wr_req<VAL_T,INDEX_T> requests[NUM_PE],
    // bram
    VAL_T bram[NUM_PE][BANK_SIZE],
    // arbitering results from read
    arbiter_result<INDEX_T> arbiter_results_from_rd[NUM_BANK]
){
    #pragma HLS inline
    // pipeline registers (_local means local to bram)
    wr_req<VAL_T,INDEX_T> requests_local[NUM_PE];
    arbiter_result<INDEX_T> arbiter_results_from_rd_local[NUM_BANK];
    #pragma HLS array_partition variable=requests_local complete
    #pragma HLS array_partition variable=arbiter_results_from_rd_local complete

    // input pipeline
    loop_wb_inpp_vpreq_unroll:
    for (unsigned int VPid = 0; VPid < NUM_PE; VPid++) {
        #pragma HLS unroll
        requests_local[VPid] =
            HLS_REG< wr_req<VAL_T,INDEX_T> >(
                HLS_REG< wr_req<VAL_T,INDEX_T> >(
                    HLS_REG< wr_req<VAL_T,INDEX_T> >(
                        HLS_REG< wr_req<VAL_T,INDEX_T> >(
                            HLS_REG< wr_req<VAL_T,INDEX_T> >(requests[VPid])))));
    }
    loop_wb_inpp_abresults_unroll:
    for (unsigned int BKid = 0; BKid < NUM_BANK; BKid++) {
        #pragma HLS unroll
        arbiter_results_from_rd_local[BKid] =
            HLS_REG< arbiter_result<INDEX_T> >(
                HLS_REG< arbiter_result<INDEX_T> >(
                    HLS_REG< arbiter_result<INDEX_T> >(
                        HLS_REG< arbiter_result<INDEX_T> >(
                            HLS_REG< arbiter_result<INDEX_T> >(arbiter_results_from_rd[BKid])))));
    }

    // write access logic
    loop_wr_logic_unroll:
    for(unsigned int BKid = 0; BKid < NUM_PE; BKid++) {
        #pragma HLS unroll
        INDEX_T vpid = arbiter_results_from_rd_local[BKid].virtual_port_id;
        if(!arbiter_results_from_rd_local[BKid].bank_idle) {
            bram[BKid][requests_local[vpid].addr >> BANK_ID_NBITS] = requests_local[vpid].data;
        }
    }
}

//----------------------------------------------------
// kernel process elements
//----------------------------------------------------
static void PE_spmspv(
    // FIFO from DL
    hls::stream<VAL_T> nnz_from_mat_stream[],
    hls::stream<INDEX_T> current_row_id_stream[],
    hls::stream<VAL_T> &nnz_from_vec_stream,
    hls::stream<INDEX_T> &current_collen_stream,
    // number of columns
    INDEX_T vec_nnz_total,
    // bram
    VAL_T bram[NUM_BANK][BANK_SIZE],
    // tile count
    unsigned int tile_cnt
) {
    // *************  PE  **************
    // PE registers
    VAL_T   nnz_from_mat[NUM_PE];
    INDEX_T current_row_id[NUM_PE];
    VAL_T   nnz_from_vec;
    VAL_T   result_inc[NUM_PE];
    INDEX_T collen;
    #pragma HLS array_partition variable=nnz_from_mat complete
    #pragma HLS array_partition variable=current_row_id complete
    #pragma HLS array_partition variable=result_inc complete
    // granted by arbiter
    bool granted[NUM_PE];
    #pragma HLS array_partition variable=granted        complete
    // whether to resend a request
    bool resend_req[NUM_PE]; // to be forwarded (also the switch to data forwarding)
    #pragma HLS array_partition variable=resend_req     complete
    // processed item count
    unsigned int fetch_cnt[NUM_PE];
    bool all_fetched[NUM_PE];
    unsigned int process_cnt[NUM_PE];
    bool all_processed[NUM_PE]; // to be forwarded
    #pragma HLS array_partition variable=fetch_cnt      complete
    #pragma HLS array_partition variable=all_fetched    complete
    #pragma HLS array_partition variable=process_cnt    complete
    #pragma HLS array_partition variable=all_processed  complete
    // forwarding signals
    INDEX_T current_row_id_fwd[NUM_PE];
    VAL_T   nnz_from_mat_fwd[NUM_PE];
    #pragma HLS array_partition variable=current_row_id_fwd  complete
    #pragma HLS array_partition variable=nnz_from_mat_fwd    complete

    // *************  Virtual Port  **************
    // no write response is needed, because if read is success, write will also success
    // Virtual port signals
    rd_req<INDEX_T> VrdP_req[NUM_PE];
    rd_resp<VAL_T> VrdP_resp[NUM_PE];
    wr_req<VAL_T,INDEX_T> VwrP_req[NUM_PE];
    #pragma HLS array_partition variable=VrdP_req complete
    #pragma HLS array_partition variable=VrdP_resp complete
    #pragma HLS array_partition variable=VwrP_req complete

    // *************  Arbiter  **************
    // save read arbiter results which will be used in write
    arbiter_result<INDEX_T> rd_arbiter_results[NUM_BANK];
    #pragma HLS array_partition variable=rd_arbiter_results complete

    // loop over all active columns
    loop_over_active_columns_PE:
    for (unsigned int vec_nnz_cnt = 0; vec_nnz_cnt < vec_nnz_total; vec_nnz_cnt++) {
        #pragma HLS pipeline off

        // used for line tracing
        #ifndef __SYNTHESIS__
            int round = 0;
            bool input_success_ltr[NUM_PE];
        #endif

        // reset PE state
        nnz_from_vec = nnz_from_vec_stream.read();
        collen = current_collen_stream.read(); // measured in number of packets
        loop_reset_process_unit_unroll:
        for (unsigned int PEid = 0; PEid < NUM_PE; PEid++) {
            nnz_from_mat[PEid] = 0;
            current_row_id[PEid] = 0;
            result_inc[PEid] = 0;

            VrdP_req[PEid].valid = 0;
            VrdP_req[PEid].zero = 0;
            VrdP_resp[PEid].valid = 0;

            granted[PEid] = false;
            resend_req[PEid] = false;

            fetch_cnt[PEid] = 0;
            all_fetched[PEid] = false;
            process_cnt[PEid] = 0;
            all_processed[PEid] = false;
        }

        // reset Arbiter results
        loop_reset_arb_results_unroll:
        for (unsigned int BKid = 0; BKid < NUM_BANK; BKid++) {
            rd_arbiter_results[BKid].virtual_port_id = 0;
            rd_arbiter_results[BKid].bank_idle = 1;
        }

        // reset priority rotation
        unsigned int rotate_priority = 0;

        // start processing
        loop_process_element_pipeline:
        while(!bool_array_and_reduction<NUM_PE>(all_fetched, all_processed)) {
            #pragma HLS pipeline II=1
            #pragma HLS latency min=9
            #pragma HLS dependence variable=bram inter RAW false
            #pragma HLS dependence variable=bram inter WAR false
            #pragma HLS dependence variable=bram inter WAW false
            // update rotate priority path
            #pragma HLS dependence variable=rotate_priority inter distance=FWD_DISTANCE     RAW true
            // data forwarding paths
            #pragma HLS dependence variable=all_processed       inter distance=FWD_DISTANCE     RAW true
            #pragma HLS dependence variable=resend_req          inter distance=FWD_DISTANCE     RAW true
            #pragma HLS dependence variable=current_row_id_fwd  inter distance=FWD_DISTANCE     RAW true
            #pragma HLS dependence variable=nnz_from_mat_fwd    inter distance=FWD_DISTANCE     RAW true

            // line tracing
            /*
            #ifndef __SYNTHESIS__
                std::cout << "Column Cnt " << vec_nnz_cnt << "\t"
                          << "Column Length " << collen << "\t"
                          << "Round " << round << "\t"
                          << "[" << array_sum<unsigned int,NUM_PE>(process_cnt) << "/" << collen * PACKET_SIZE << "]"
                          << std::endl;
                round ++;
                if(round > MAX_SW_EMU_LIMIT) {
                    std::cout << "[ERROR] Exceeded max software emulation loop count. Probably there is a deadlock" << std::endl;
                    return;
                }
            #endif
            */

            // fetch inputs and send requests
            loop_pe_fetch_unroll:
            for (unsigned int PEid = 0; PEid < NUM_PE; PEid++) {
                #pragma HLS unroll
                // all_fetched
                all_fetched[PEid] = (fetch_cnt[PEid] >= collen);

                // need to fetch or resend?
                if(!all_fetched[PEid] || !all_processed[PEid]) {
                    // if we need to resend a request
                    if(resend_req[PEid]) {
                        // use forwarded values of A,B and C from post_arbiter stage. Do not read from DL
                        current_row_id[PEid] = current_row_id_fwd[PEid];
                        nnz_from_mat[PEid] = nnz_from_mat_fwd[PEid];
                        VrdP_req[PEid].addr = current_row_id_fwd[PEid];
                        VrdP_req[PEid].zero = (nnz_from_mat_fwd[PEid] == 0);
                        VrdP_req[PEid].valid = true;
                    } else { // else : normal fetch
                        // need to fetch
                        if(!all_fetched[PEid]) {
                            INDEX_T cri;
                            VAL_T   nfm;
                            bool cri_rdsuccess = current_row_id_stream[PEid].read_nb(cri);
                            bool nfm_rdsuccess = nnz_from_mat_stream[PEid].read_nb(nfm);
                            // fetch success
                            if(cri_rdsuccess && nfm_rdsuccess) {
                                cri -= tile_cnt * TILE_SIZE;
                                current_row_id[PEid] = cri;
                                nnz_from_mat[PEid] = nfm;
                                VrdP_req[PEid].valid = true;
                                VrdP_req[PEid].addr = cri;
                                VrdP_req[PEid].zero = (nfm == 0);
                                fetch_cnt[PEid] ++;
                            } else { // fetch failed
                                current_row_id[PEid] = 0;
                                nnz_from_mat[PEid] = 0;
                                VrdP_req[PEid].valid = false;
                                VrdP_req[PEid].addr = 0;
                                VrdP_req[PEid].zero = false;
                            }

                            // used for line tracing
                            // #ifndef __SYNTHESIS__
                            //     input_success_ltr[PEid] = cri_rdsuccess && nfm_rdsuccess;
                            // #endif

                        } else { // no need to fetch
                            current_row_id[PEid] = 0;
                            nnz_from_mat[PEid] = 0;
                            VrdP_req[PEid].valid = false;
                            VrdP_req[PEid].addr = 0;
                            VrdP_req[PEid].zero = false;
                        }
                    }
                } else { // no need to either fetch or resend
                    current_row_id[PEid] = 0;
                    nnz_from_mat[PEid] = 0;
                    VrdP_req[PEid].valid = false;
                    VrdP_req[PEid].addr = 0;
                    VrdP_req[PEid].zero = false;
                }
            }
            // ------------ end of F stage

            // arbiter logic
            bram_access_arbiter(
                VrdP_req,
                rd_arbiter_results,
                granted,
                rotate_priority
            );
            // ------------ end of A stage

            // forward ungranted requests to F stage, update process_cnt
            loop_pe_postarb_unroll:
            for (unsigned int PEid = 0; PEid < NUM_PE; PEid++) {
                #pragma HLS unroll
                // these signals need to be forwarded to the fitst stage;
                all_processed[PEid] = (process_cnt[PEid] >= collen);

                // activate data forwarding when a request fails
                if(VrdP_req[PEid].valid && (!granted[PEid]) && (!VrdP_req[PEid].zero)) {
                    current_row_id_fwd[PEid] = current_row_id[PEid];
                    nnz_from_mat_fwd[PEid] = nnz_from_mat[PEid];
                    resend_req[PEid] = true;
                } else {
                    resend_req[PEid] = false;
                }

                // count (granted, valid) or (zero) requests as processed
                if((VrdP_req[PEid].valid && granted[PEid]) ||
                    (VrdP_req[PEid].valid && VrdP_req[PEid].zero)) {
                    process_cnt[PEid] ++;
                }
            }
            // update priority rotate
            rotate_priority = (rotate_priority + 1) % NUM_PE;
            // ------------ end of P stage

            // process read requests
            bram_access_read(VrdP_req,VrdP_resp,bram,rd_arbiter_results,granted);
            // ------------ end of R stage

            // execute and send write requests
            loop_process_element_X_unroll:
            for(unsigned int PEid = 0; PEid < NUM_PE; PEid++) {
                #pragma HLS unroll
                if(VrdP_resp[PEid].valid && (nnz_from_mat[PEid] != 0)) {
#if defined(MulAddSemiring)
                    result_inc[PEid] = nnz_from_mat[PEid] * nnz_from_vec;
                    VwrP_req[PEid].data = VrdP_resp[PEid].data + result_inc[PEid];
                    VwrP_req[PEid].addr = current_row_id[PEid] - tile_cnt * TILE_SIZE;
#elif defined(LogicalAndOrSemiring)
                    result_inc[PEid] = nnz_from_mat[PEid] && nnz_from_vec;
                    VwrP_req[PEid].data = VrdP_resp[PEid].data || result_inc[PEid];
                    VwrP_req[PEid].addr = current_row_id[PEid] - tile_cnt * TILE_SIZE;
#else
                    std::cout << "Invalid semiring" << std::endl;
                    exit(EXIT_FAILURE);
#endif
                } else {
                    result_inc[PEid] = 0;
                    VwrP_req[PEid].data = 0;
                    VwrP_req[PEid].addr = 0;
                }
            }
            // ------------ end of X stage

            // process write requests
            bram_access_write(VwrP_req,bram,rd_arbiter_results);
            // ------------ end of W stage

            // line tracing
            /*
            #ifndef __SYNTHESIS__
                // PE states
                for (size_t k = 0; k < NUM_PE; k++) {
                    bool display_value = input_success_ltr[k] || !granted[k];
                    std::cout << "PE [" << std::setw(2) << k << "] {"
                                        << ""  << std::setw(4) << (all_processed[k]     ?  "----" :  std::to_string(process_cnt[k]))                << ""  << ""
                                        << "} {"
                                        << ""  << std::setw(2) << (input_success_ltr[k] ? "->" : "--")                                              << ""  << "|"
                                        << ""  << std::setw(9) << (display_value        ? std::to_string((float)nnz_from_mat[k])            : "--") << ""  << "|"
                                        << ""  << std::setw(9) << (display_value        ? std::to_string((float)nnz_from_vec)               : "--") << ""  << "|"
                                        << "[" << std::setw(5) << (display_value        ? std::to_string(current_row_id[k])                 : "--") << ""  << ""
                                        << "(" << std::setw(2) << (display_value        ? std::to_string(current_row_id[k] & BANK_ID_MASK)  : "--") << ")]"<< "|"
                                        << ""  << std::setw(9) << (VrdP_resp[k].valid   ? std::to_string((float)VrdP_resp[k].data)          : "--") << ""  << "|"
                                        << ""  << std::setw(9) << (VrdP_resp[k].valid   ? std::to_string((float)VwrP_req[k].data)           : "--") << ""
                                        << "} {"
                                        << (VrdP_req[k].zero    ? "Rz" :
                                                VrdP_req[k].valid   ? "Rv" : " ." ) << ":"
                                        << (!granted[k]         ?  "x" :  "o" ) << "|"
                                        << (VrdP_resp[k].valid  ? "Wc" : " ." ) << "|"
                                        << "}";
                    std::cout << std::endl;
                }

                // Arbiter results
                std::cout << "BA (R" << std::setw(2) << ((rotate_priority- 1) % NUM_PE) << ") {";
                for (size_t x = 0; x < NUM_PE; x++) {
                    std::cout << "["
                              << ""  << std::setw(2) << x << ""  << ":"
                              << ""  << std::setw(2) << ((rd_arbiter_results[x].bank_idle) ? "--" : std::to_string(rd_arbiter_results[x].virtual_port_id)) << ""  << ""
                              << "]";
                }
                std::cout << "}";
                std::cout << std::endl;

            #endif
            */
        }
    }
}

//----------------------------------------------------
// kernel execution
//----------------------------------------------------
static void execution_spmspv(
    // gmem pointer
    const PACKED_DWI_T mat_dwi_ddr[],
    const INDEX_T mat_idxptr_ddr[],
    const DIT_T vec_dit_ddr[],
    // size
    unsigned int vec_nnz_total,
    // bram
    VAL_T bram[NUM_BANK][BANK_SIZE],
    // tiling parameters
    const unsigned int num_columns,
    unsigned int tile_cnt,
    unsigned int tile_base
) {
    // FIFOs
    static hls::stream<INDEX_T> current_collen_stream;
    #pragma HLS STREAM variable=current_collen_stream depth=256

    static hls::stream<INDEX_T> current_row_id_stream[NUM_PE];
    #pragma HLS STREAM variable=current_row_id_stream depth=256

    static hls::stream<VAL_T> nnz_from_mat_stream[NUM_PE];
    #pragma HLS STREAM variable=nnz_from_mat_stream depth=256

    static hls::stream<VAL_T> nnz_from_vec_stream;
    #pragma HLS STREAM variable=nnz_from_vec_stream depth=256

    #pragma HLS dataflow

    DL_spmspv(
        mat_dwi_ddr,
        mat_idxptr_ddr,
        vec_dit_ddr,
        nnz_from_mat_stream,
        current_row_id_stream,
        nnz_from_vec_stream,
        current_collen_stream,
        vec_nnz_total,
        num_columns,
        tile_cnt,
        tile_base
    );

    #ifndef __SYNTHESIS__
        std::cout << "[INFO kernel_spmspv] DL complete" << std::endl;
        std::cout.flush();
    #endif

    PE_spmspv(
        nnz_from_mat_stream,
        current_row_id_stream,
        nnz_from_vec_stream,
        current_collen_stream,
        vec_nnz_total,
        bram,
        tile_cnt
    );

    #ifndef __SYNTHESIS__
        std::cout << "[INFO kernel_spmspv] PE complete" << std::endl;
        std::cout.flush();
    #endif
}

//----------------------------------------------------
// change results to sparse
//----------------------------------------------------

static void checkout_results(
    // data to be checked
    VAL_T dense_data[NUM_BANK][BANK_SIZE],
    // FIFOs
    hls::stream<VAL_T> nnz_streams[],
    hls::stream<INDEX_T> idx_streams[],
    // control signals
    hls::stream<unsigned int> &Nnnz_no_mask
) {
    unsigned int local_Nnnz_no_mask = 0;
    loop_over_dense_data_pipeline:
    for (unsigned int round_cnt = 0; round_cnt < BANK_SIZE / NUM_PORT_PER_BANK; round_cnt ++) {
        #pragma HLS pipeline II=1
        unsigned int local_Nnnz_inc = 0;

        loop_banks_unroll:
        for (unsigned int Bank_id = 0; Bank_id < NUM_BANK; Bank_id++) {
            #pragma HLS unroll
            loop_ports_unroll:
            for (unsigned int Port_id = 0; Port_id < NUM_PORT_PER_BANK; Port_id++) {
                #pragma HLS unroll
                INDEX_T index = Bank_id + (round_cnt * NUM_PORT_PER_BANK + Port_id) * NUM_BANK;
                VAL_T data = 0;
                // pipeline the control signal
                bool BRAM_RD_ENABLE_local =
                    HLS_REG<bool>(
                        HLS_REG<bool>(
                            HLS_REG<bool>(
                                HLS_REG<bool>(
                                    HLS_REG<bool>(round_cnt < BANK_SIZE / NUM_PORT_PER_BANK)))));
                if(BRAM_RD_ENABLE_local) {
                    // input pipeline to BRAM
                    INDEX_T bank_id_bram_local =
                        HLS_REG<INDEX_T>(
                            HLS_REG<INDEX_T>(
                                HLS_REG<INDEX_T>(
                                    HLS_REG<INDEX_T>(
                                        HLS_REG<INDEX_T>(Bank_id)))));
                    INDEX_T sub_bank_addr_bram_local =
                        HLS_REG<INDEX_T>(
                                HLS_REG<INDEX_T>(
                                    HLS_REG<INDEX_T>(
                                        HLS_REG<INDEX_T>(
                                            HLS_REG<INDEX_T>(round_cnt * NUM_PORT_PER_BANK + Port_id)))));
                    // output pipeline from BRAM
                    data =
                        HLS_REG<VAL_T>(
                            HLS_REG<VAL_T>(
                                HLS_REG<VAL_T>(
                                    HLS_REG<VAL_T>(
                                        HLS_REG<VAL_T>(dense_data[bank_id_bram_local][sub_bank_addr_bram_local])))));
                }
                if(data) {
                    nnz_streams[Bank_id * NUM_PORT_PER_BANK + Port_id].write(data);
                    idx_streams[Bank_id * NUM_PORT_PER_BANK + Port_id].write(index);
                    local_Nnnz_inc ++;
                }
            }
        }

        local_Nnnz_no_mask += local_Nnnz_inc;
    }
    Nnnz_no_mask << local_Nnnz_no_mask;
}

//----------------------------------------------------
// write back to ddr (out-of-order)
//----------------------------------------------------

static void write_back_ddr(
    // FIFOs
    hls::stream<VAL_T> nnz_streams[],
    hls::stream<INDEX_T> idx_streams[],
    // positive mask. valid when mask[i] != 0.
#if defined(USE_MASK)
    const INDEX_T mask[],
#endif
    DIT_T sparse_dit[],
    // control signals
    hls::stream<unsigned int> &Nnnz_no_mask,
    // count results
    unsigned int &Nnnz,
    // tile count
    unsigned int tile_cnt
) {
    unsigned int wb_cnt = 0;
    unsigned int local_Nnnz_no_mask;
    bool checkout_finish = false;

    // used forline tracing
    #ifndef __SYNTHESIS__
        unsigned int round = 0;
    #endif
    loop_until_all_written_back:
    while(!(checkout_finish && (local_Nnnz_no_mask == wb_cnt))) {
        if(!checkout_finish) { // only try to read when checkout is not finished (avoid corrputing local_Nnnz_no_mask)
            checkout_finish |= Nnnz_no_mask.read_nb(local_Nnnz_no_mask); // once successfully read, set up flag
        }

        loop_over_checkout_lanes_pipeline:
        for (unsigned int Lane_id = 0; Lane_id < NUM_LANE; Lane_id++) {
            #pragma HLS pipeline II=1
            DIT_T wb_temp;
            if(idx_streams[Lane_id].read_nb(wb_temp.index)) {
                nnz_streams[Lane_id].read_nb(wb_temp.data);
                wb_temp.index += tile_cnt * TILE_SIZE;
                wb_cnt ++;

                // positive mask. valid when mask[i] != 0.
#if defined(USE_MASK)
    #if defined(MASK_WRITE_TO_ZERO)
                if(mask[wb_temp.index] == 0)
    #endif
    #if defined(MASK_WRITE_TO_ONE)
                if(mask[wb_temp.index] != 0)
    #endif
#endif
                {
                    sparse_dit[Nnnz + 1] = wb_temp;
                    Nnnz ++;

//                     line tracing
//                     #ifndef __SYNTHESIS__
//                     std::cout << "["   << wb_cnt << "/" << local_Nnnz_no_mask << "]";
//                     std::cout << "LN"  << std::setw(2) << Lane_id << " : "
//                               << "{("  << std::setw(5) << wb_temp.data  << ", "
//                               << "@"   << std::setw(5) << wb_temp.index << ")|"
// #if defined(USE_MASK)
//                               << "M "  << std::setw(1) << mask[wb_temp.index]
// #endif
//                               << " >> B["  << std::setw(5) << Nnnz << "]}"
//                               << std::endl  << std::flush;
// #endif
                }
            }
        }
    }
}

//----------------------------------------------------
// kernel result write back
//----------------------------------------------------

static void write_back_spmspv(
    VAL_T bram[NUM_BANK][BANK_SIZE],
    DIT_T results_ddr[],
#if defined(USE_MASK)
    const INDEX_T mask_ddr[],
#endif
    // result non-zero count(after mask)
    unsigned int &result_nnz_cnt,
    // tile count
    unsigned int tile_cnt
){
    static hls::stream<VAL_T> nnz_streams[NUM_LANE];
    static hls::stream<INDEX_T> idx_streams[NUM_LANE];
    static hls::stream<unsigned int> Nnnz_no_mask;
    #pragma HLS array_partition variable=nnz_streams complete
    #pragma HLS array_partition variable=idx_streams complete
    #pragma HLS stream variable=nnz_streams depth=32
    #pragma HLS stream variable=idx_streams depth=32
    #pragma HLS stream variable=Nnnz_no_mask depth=1

    #pragma HLS dataflow
    checkout_results(
        bram,
        nnz_streams,
        idx_streams,
        Nnnz_no_mask
    );

    #ifndef __SYNTHESIS__
        std::cout << "[INFO kernel_spmspv] CHECKOUT complete" << std::endl;
        std::cout.flush();
    #endif

    write_back_ddr(
        nnz_streams,
        idx_streams,
#if defined(USE_MASK)
        mask_ddr,
#endif
        results_ddr,
        Nnnz_no_mask,
        result_nnz_cnt,
        tile_cnt
    );

    #ifndef __SYNTHESIS__
        std::cout << "[INFO kernel_spmspv] WRITE DDR complete" << std::endl;
        std::cout.flush();
    #endif
}

//----------------------------------------------------
// kernel spmspv
//----------------------------------------------------
extern "C" {

void kernel_spmspv(
    // ddr pointer
    const DIT_T* vec_dit_ddr,
    const PACKED_DWI_T* mat_dwi_ddr,
    const INDEX_T* mat_idxptr_ddr,
    const INDEX_T* mat_tileptr_ddr,
#if defined(USE_MASK)
    const INDEX_T* mask_ddr,
#endif
    DIT_T* result_ddr,
    // size
    const unsigned int num_columns,
    const unsigned int num_tiles
) {

    // interfaces
    #pragma HLS INTERFACE m_axi port=mat_dwi_ddr      offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=mat_idxptr_ddr   offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=mat_tileptr_ddr  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=vec_dit_ddr      offset=slave bundle=gmem2
#if defined(USE_MASK)
    #pragma HLS INTERFACE m_axi port=mask_ddr         offset=slave bundle=gmem2
#endif
    #pragma HLS INTERFACE m_axi port=result_ddr       offset=slave bundle=gmem3

    #pragma HLS INTERFACE s_axilite port=mat_dwi_ddr      bundle=control
    #pragma HLS INTERFACE s_axilite port=mat_idxptr_ddr   bundle=control
    #pragma HLS INTERFACE s_axilite port=mat_tileptr_ddr  bundle=control
    #pragma HLS INTERFACE s_axilite port=vec_dit_ddr      bundle=control
#if defined(USE_MASK)
    #pragma HLS INTERFACE s_axilite port=mask_ddr         bundle=control
#endif
    #pragma HLS INTERFACE s_axilite port=result_ddr       bundle=control

    #pragma HLS INTERFACE s_axilite port=num_columns      bundle=control
    #pragma HLS INTERFACE s_axilite port=num_tiles        bundle=control

    #pragma HLS INTERFACE s_axilite port=return           bundle=control

    #pragma HLS data_pack variable=mat_dwi_ddr
    #pragma HLS data_pack variable=vec_dit_ddr
    #pragma HLS data_pack variable=result_ddr

    // block ram (output buffer)
    static VAL_T bram[NUM_BANK][BANK_SIZE];
    #pragma HLS array_partition variable=bram complete dim=1
    // *************  Pipelining BRAM ***************
    #pragma HLS resource variable=bram core=RAM_2P_BRAM latency=4

    unsigned int result_nnz_cnt_localreg = 0;
    unsigned int vec_nnz_total = vec_dit_ddr[0].index;

    // line tracing
    #ifndef __SYNTHESIS__
        std::cout << "[INFO kernel_spmspv] start, input vector non-zero count : " << vec_nnz_total << std::endl;
    #endif

    // loop over all tiles
    loop_over_all_tiles:
    for (unsigned int tile_cnt = 0; tile_cnt < num_tiles; tile_cnt++) {
        #pragma HLS loop_flatten off
        #pragma HLS pipeline off

        // line tracing
        #ifndef __SYNTHESIS__
            std::cout << "[INFO kernel_spmspv] Tile " << std::setw(2) << tile_cnt << " Buffer reset" << std::endl;
        #endif

        // reset output buffer
        loop_reset_output_buffer:
        for(unsigned int i = 0; i < BANK_SIZE / NUM_PORT_PER_BANK; i++) {
            #pragma HLS pipeline II=1
            loop_reset_ob_bank_unroll:
            for(unsigned int j = 0; j < NUM_BANK; j++) {
                #pragma HLS unroll
                loop_reset_ob_port_unroll:
                for (unsigned int p = 0; p < NUM_PORT_PER_BANK; p++) {
                    #pragma HLS unroll
                    bram[j][i * NUM_PORT_PER_BANK + p] = 0;
                }
            }
        }

        // read tile base
        INDEX_T tile_base = mat_tileptr_ddr[tile_cnt];

        // line tracing
        #ifndef __SYNTHESIS__
            std::cout << "[INFO kernel_spmspv] Tile " << std::setw(2) << tile_cnt
                                << " Base at " << std::setw(5) << tile_base << std::endl;
        #endif

        // execution
        execution_spmspv(
            mat_dwi_ddr,
            mat_idxptr_ddr,
            vec_dit_ddr,
            vec_nnz_total,
            bram,
            num_columns,
            tile_cnt,
            tile_base
        );

        // line tracing
        #ifndef __SYNTHESIS__
            std::cout << "[INFO kernel_spmspv] Tile " << std::setw(2) << tile_cnt << " EX complete" << std::endl;
        #endif

        // write back to result_ddr
        write_back_spmspv(
            bram,
            result_ddr,
#if defined(USE_MASK)
            mask_ddr,
#endif
            result_nnz_cnt_localreg,
            tile_cnt
        );

        // line tracing
        #ifndef __SYNTHESIS__
            std::cout << "[INFO kernel_spmspv] Tile " << std::setw(2) << tile_cnt << " WB complete" << std::endl;
        #endif

    } // loop over all tiles

    // report nnz count
    DIT_T result_head;
    result_head.index = result_nnz_cnt_localreg;
    result_head.data = 0;
    result_ddr[0] = result_head;

    // line tracing
    #ifndef __SYNTHESIS__
        std::cout << "[INFO kernel_spmspv] finish, result non-zero count : " << result_nnz_cnt_localreg << std::endl;
    #endif
}

} // extern "C"
