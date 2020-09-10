#include <ap_fixed.h>
#include <assert.h>
#include <iostream>

#include "./kernel_assign_vector_sparse.h"

extern "C" {

void kernel_assign_vector_sparse(
    const VI_T *mask,          // The sparse mask vector. The index field of the first element is the length.
    VAL_T *inout,              // The inout vector
    VI_T  *new_frontier,       // The new frontier. The index field of the first element is the length.
    const unsigned int  mode,  // Working mode. 0 for BFS, 1 for SSSP
    VAL_T val                  // The value to be assigned to the inout vector
) {
    /*
    *   working mode description:
    *   0 : used in BFS.
    *       the new_frontier will never be updated.
    *       the inout will be assigned val according to the mask.
    *   1 : used in SSSP.
    *       the input val will never be used.
    *       the inout will be updated to have the smaller value between the inout and mask.
    *       new_frontier will be generated
    */
#pragma HLS INTERFACE m_axi port=mask offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=inout offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=new_frontier offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=mask bundle=control
#pragma HLS INTERFACE s_axilite port=inout bundle=control
#pragma HLS INTERFACE s_axilite port=new_frontier bundle=control

#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATA_PACK variable=mask
#pragma HLS DATA_PACK variable=new_frontier


    // local buffer
    VAL_T   local_inout_buf[BATCH_SIZE];
    VI_T    local_mask_buf[BATCH_SIZE];
    VI_T    local_nf_buf[BATCH_SIZE];
    #pragma HLS DATA_PACK variable=local_mask_buf
    #pragma HLS DATA_PACK variable=local_nf_buf

    INDEX_T length = mask[0].index;
    unsigned int num_batches = (length + BATCH_SIZE - 1) / BATCH_SIZE;
    unsigned int remain = length;

    unsigned int nf_length = 0;

    loop_over_batches:
    for (unsigned int batch_cnt = 0; batch_cnt < num_batches; batch_cnt++) {
        #pragma HLS pipeline off

        unsigned int batch_nf_length = 0;

        // read stage
        loop_read_inout_val:
        for (unsigned int i = 0; i < BATCH_SIZE; i++) {
            #pragma HLS pipeline II=1
            if(i < remain) {
                INDEX_T midx = mask[i + 1 + batch_cnt * BATCH_SIZE].index;
                VAL_T mval = mask[i + 1 + batch_cnt * BATCH_SIZE].val;
                local_mask_buf[i].index = midx;
                local_mask_buf[i].val = mval;
                local_inout_buf[i] = inout[midx];
            }
        }
        #ifndef __SYNTHESIS__
            std::cout << "[INFO kernel_assign_vector_sparse] Batch " << batch_cnt <<" read complete" << std::endl << std::flush;
        #endif

        // process stage
        loop_process:
        for (unsigned int i = 0; i < BATCH_SIZE; i++) {
            #pragma HLS pipeline II=1
            if(i < remain) {
                switch (mode) {
                case 0: // BFS
                    local_inout_buf[i] = val;
                    break;
                case 1: // SSSP
                    if(local_inout_buf[i] > local_mask_buf[i].val) {
                        // #ifndef __SYNTHESIS__
                        //     std::cout << "[INFO kernel_assign_vector_sparse] Update at "
                        //         << local_mask_buf[i].index << " "
                        //         << "(" << local_inout_buf[i] << " > " << local_mask_buf[i].val << ")" << std::endl << std::flush;
                        // #endif
                        local_inout_buf[i] = local_mask_buf[i].val;
                        local_nf_buf[batch_nf_length] = local_mask_buf[i];
                        batch_nf_length ++;
                    }
                    break;
                default:
                    #ifndef __SYNTHESIS__
                        std::cout << "Invalid mask type" << std::endl;
                    #endif
                    break;
                }
            }
        }
        #ifndef __SYNTHESIS__
            std::cout << "[INFO kernel_assign_vector_sparse] Batch " << batch_cnt <<" process complete" << std::endl << std::flush;
        #endif

        // wirte inout
        loop_write_inout_val:
        for (unsigned int i = 0; i < BATCH_SIZE; i++) {
            #pragma HLS pipeline II=1
            if(i < remain) {
                inout[local_mask_buf[i].index] = local_inout_buf[i];
            }
        }

        // wirte new_frontier
        if(mode == 1) {
            loop_write_new_frontier:
            for (unsigned int i = 0; i < batch_nf_length; i++) {
                #pragma HLS pipeline II=1
                new_frontier[i + 1 + nf_length] = local_nf_buf[i];
            }
            nf_length += batch_nf_length;
        }
        #ifndef __SYNTHESIS__
            std::cout << "[INFO kernel_assign_vector_sparse] Batch " << batch_cnt <<" write complete" << std::endl << std::flush;
        #endif

        // update progress
        remain -= BATCH_SIZE;
    }

    // attach head to new_frontier
    if(mode == 1) {
        VI_T nf_head;
        nf_head.index = nf_length;
        nf_head.val = 0;
        new_frontier[0] = nf_head;
    }

}

} // extern "C"
