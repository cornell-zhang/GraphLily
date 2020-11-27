#include "./kernel_apply.h"

#include <assert.h>
#include <iostream>
#include <cstdlib>


// TODO(@Yixiao): Batch the loop to avoid reading and writing inout at the same iteration.

void kernel_assign_vector_dense(
    const PACKED_VAL_T *mask,  // The mask vector
    PACKED_VAL_T *inout,       // The inout vector
    unsigned length,           // The length of the mask/inout vector
    VAL_T val,                 // The value to be assigned to the inout vector
    MASK_T mask_type           // The mask type
) {
    assert(length % PACK_SIZE == 0);
    unsigned size = length / PACK_SIZE;
    PACKED_VAL_T tmp_mask;
    PACKED_VAL_T tmp_inout;

    loop_kernel_assign_vector_dense:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS dependence variable=inout inter false

        tmp_mask = mask[i];
        tmp_inout = inout[i];
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
        inout[i] = tmp_inout;
    }
}
