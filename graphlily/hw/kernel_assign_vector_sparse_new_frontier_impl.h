#include "libfpga/hisparse.h"


static void kernel_assign_vector_sparse_new_frontier(
    const IDX_VAL_T *mask,    // The sparse mask vector. The index field of the first element is the length.
    VAL_T *inout,             // The inout vector.
    IDX_VAL_T *new_frontier   // The new frontier. The index field of the first element is the length.
) {
    // local buffer
    VAL_T local_inout_buf[BATCH_SIZE];
    IDX_VAL_T local_mask_buf[BATCH_SIZE];
    IDX_VAL_T local_new_frontier_buf[BATCH_SIZE];
    #pragma HLS DATA_PACK variable=local_mask_buf
    #pragma HLS DATA_PACK variable=local_new_frontier_buf

    IDX_T length = mask[0].index;
    unsigned num_batches = (length + BATCH_SIZE - 1) / BATCH_SIZE;
    unsigned remain = length;

    unsigned new_frontier_length = 0;

    loop_over_batches:
    for (unsigned batch_cnt = 0; batch_cnt < num_batches; batch_cnt++) {
        #pragma HLS pipeline off
        unsigned batch_new_frontier_length = 0;

        // read stage
        loop_read_inout_val:
        for (unsigned i = 0; i < BATCH_SIZE; i++) {
            #pragma HLS pipeline II=1
            if (i < remain) {
                IDX_VAL_T tmp_mask = mask[i + 1 + batch_cnt * BATCH_SIZE];
                local_mask_buf[i].index = tmp_mask.index;
                local_mask_buf[i].val = tmp_mask.val;
                local_inout_buf[i] = inout[tmp_mask.index];
            }
        }

        // process stage
        loop_process:
        for (unsigned i = 0; i < BATCH_SIZE; i++) {
            #pragma HLS pipeline II=1
            if (i < remain) {
                if (local_inout_buf[i] > local_mask_buf[i].val) {
                    local_inout_buf[i] = local_mask_buf[i].val;
                    local_new_frontier_buf[batch_new_frontier_length] = local_mask_buf[i];
                    batch_new_frontier_length++;
                }
            }
        }

        // write inout
        loop_write_inout_val:
        for (unsigned i = 0; i < BATCH_SIZE; i++) {
            #pragma HLS pipeline II=1
            if (i < remain) {
                inout[local_mask_buf[i].index] = local_inout_buf[i];
            }
        }

        // write new_frontier
        loop_write_new_frontier:
        for (unsigned i = 0; i < batch_new_frontier_length; i++) {
            #pragma HLS pipeline II=1
            new_frontier[i + 1 + new_frontier_length] = local_new_frontier_buf[i];
        }
        new_frontier_length += batch_new_frontier_length;

        // update progress
        remain -= BATCH_SIZE;
    }

    // attach head to new_frontier
    IDX_VAL_T new_frontier_head;
    new_frontier_head.index = new_frontier_length;
    new_frontier_head.val = 0;
    new_frontier[0] = new_frontier_head;
}
