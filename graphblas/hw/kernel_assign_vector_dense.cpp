#include <assert.h>

#include "./kernel_assign_vector_dense.h"


extern "C" {

void kernel_assign_vector_dense(
    const VECTOR_T *mask,  // The mask vector
    VECTOR_T *inout,       // The inout vector
    unsigned int length,   // The length of the mask/inout vector
    VECTOR_T val           // The value to be assigned to the inout vector
) {
#pragma HLS INTERFACE m_axi port=mask offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=inout offset=slave bundle=gmem1

#pragma HLS INTERFACE s_axilite port=mask bundle=control
#pragma HLS INTERFACE s_axilite port=inout bundle=control

#pragma HLS INTERFACE s_axilite port=length bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    VECTOR_T tmp_mask;

    loop_kernel_assign_vector_dense:
    for (int i = 0; i < length; i++) {
        #pragma HLS PIPELINE II=1
        tmp_mask = mask[i];
#if defined(MASK_WRITE_TO_ZERO)
        if (tmp_mask == 0) {
            inout[i] = val;
        }
#elif defined(MASK_WRITE_TO_ONE)
        if (tmp_mask != 0) {
            inout[i] = val;
        }
#else
        std::cout << "Invalid mask type" << std::endl;
        exit(EXIT_FAILURE);
#endif
    }
}

} // extern "C"
