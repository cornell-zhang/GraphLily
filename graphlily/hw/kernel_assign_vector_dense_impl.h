#include "./overlay.h"

#include <assert.h>
#include <iostream>
#include <cstdlib>


void kernel_assign_vector_dense(
    const PACKED_VAL_T *mask,  // The mask vector
    PACKED_VAL_T *inout,       // The inout vector
    unsigned length,           // The length of the mask/inout vector
    VAL_T val,                 // The value to be assigned to the inout vector
    MASK_T mask_type           // The mask type
) {
    assert(length % PACK_SIZE == 0);
    unsigned size = length / PACK_SIZE;

    // local buffer
    PACKED_VAL_T local_mask_buf[BATCH_SIZE];
    PACKED_VAL_T local_inout_buf[BATCH_SIZE];
    #pragma HLS DATA_PACK variable=local_mask_buf
    #pragma HLS DATA_PACK variable=local_inout_buf

    unsigned num_batches = (size + BATCH_SIZE - 1) / BATCH_SIZE;
    unsigned remain = size;

    loop_over_batches:
    for (unsigned batch_cnt = 0; batch_cnt < num_batches; batch_cnt++) {
        #pragma HLS pipeline off

        // read stage
        loop_read_mask_and_inout:
        for (unsigned i = 0; i < BATCH_SIZE; i++) {
            #pragma HLS pipeline II=1
            if (i < remain) {
                PACKED_VAL_T tmp_mask = mask[i + batch_cnt * BATCH_SIZE];
                local_mask_buf[i] = tmp_mask;
                PACKED_VAL_T tmp_inout = inout[i + batch_cnt * BATCH_SIZE];
                local_inout_buf[i] = tmp_inout;
            }
        }

        // process stage
        loop_process:
        for (unsigned i = 0; i < BATCH_SIZE; i++) {
            #pragma HLS pipeline II=1
            #pragma HLS dependence variable=local_inout_buf inter false

            PACKED_VAL_T tmp_mask = local_mask_buf[i];
            PACKED_VAL_T tmp_inout = local_inout_buf[i];
            for (int k = 0; k < PACK_SIZE; k++) {
                #pragma HLS UNROLL
                if (mask_type == WRITETOZERO) {
                    if (tmp_mask.data[k] == 0) {
                        tmp_inout.data[k] = val;
                    }
                } else if (mask_type == WRITETOONE) {
                    if (tmp_mask.data[k] != 0) {
                        tmp_inout.data[k] = val;
                    }
                } else {
                    std::cout << "Invalid mask type" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
            local_inout_buf[i] = tmp_inout;
        }

        // write inout
        loop_write_inout:
        for (unsigned i = 0; i < BATCH_SIZE; i++) {
            #pragma HLS pipeline II=1
            if (i < remain) {
                inout[i + batch_cnt * BATCH_SIZE] = local_inout_buf[i];
            }
        }

        // update progress
        remain -= BATCH_SIZE;
    }
}
