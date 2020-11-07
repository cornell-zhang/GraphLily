#include "./kernel_assign_vector_dense.h"

#include <assert.h>
#include <iostream>

#include <ap_fixed.h>


extern "C" {

void kernel_assign_vector_dense(
    const PACKED_VAL_T *mask,  // The mask vector
    PACKED_VAL_T *inout,       // The inout vector
    unsigned length,           // The length of the mask/inout vector
    VAL_T val                  // The value to be assigned to the inout vector
) {
#pragma HLS INTERFACE m_axi port=mask offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=inout offset=slave bundle=gmem0

#pragma HLS INTERFACE s_axilite port=mask bundle=control
#pragma HLS INTERFACE s_axilite port=inout bundle=control

#pragma HLS INTERFACE s_axilite port=length bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATA_PACK variable=mask
#pragma HLS DATA_PACK variable=inout

    assert(length % PACK_SIZE == 0);
    unsigned size = length / PACK_SIZE;
    PACKED_VAL_T tmp_mask;
    PACKED_VAL_T tmp_inout;

    loop_kernel_assign_vector_dense:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        tmp_mask = mask[i];
        tmp_inout = inout[i];
        for (int k = 0; k < PACK_SIZE; k++) {
            #pragma HLS UNROLL
#if defined(MASK_WRITE_TO_ZERO)
            if (tmp_mask.data[k] == 0) {
                tmp_inout.data[k] = val;
            }
#elif defined(MASK_WRITE_TO_ONE)
            if (tmp_mask.data[k] != 0) {
                tmp_inout.data[k] = val;
            }
#else
        std::cout << "Invalid mask type" << std::endl;
        exit(EXIT_FAILURE);
#endif
        }
        inout[i] = tmp_inout;
    }
}

}  // extern "C"
