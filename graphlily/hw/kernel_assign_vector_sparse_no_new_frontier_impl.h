#include "./overlay.h"


void kernel_assign_vector_sparse_no_new_frontier(
    const IDX_VAL_T *mask,  // The sparse mask vector. The index field of the first element is the length.
    VAL_T *inout,           // The inout vector.
    VAL_T val,              // The value to be assigned to the inout vector.
    bool enable             // Kernel enable
) {
    if (!enable) return;

    // local buffer
    VAL_T local_inout_buf[BATCH_SIZE];
    IDX_VAL_T local_mask_buf[BATCH_SIZE];
    #pragma HLS DATA_PACK variable=local_mask_buf

    IDX_T length = mask[0].index;
    unsigned num_batches = (length + BATCH_SIZE - 1) / BATCH_SIZE;
    unsigned remain = length;

    loop_over_batches:
    for (unsigned batch_cnt = 0; batch_cnt < num_batches; batch_cnt++) {
        #pragma HLS pipeline off

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
                local_inout_buf[i] = val;
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

        // update progress
        remain -= BATCH_SIZE;
    }
}
