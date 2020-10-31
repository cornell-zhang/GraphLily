#include <ap_fixed.h>
#include <assert.h>
#include <iostream>

#include "./kernel_add_scalar_vector_dense.h"


extern "C" {

void kernel_add_scalar_vector_dense(
    const PACKED_VAL_T *in,  // The input vector
    PACKED_VAL_T *out,       // The output vector
    unsigned length,         // The length of the in/out vector
    VAL_T val                // The value to be added
) {
#pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem0

#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control

#pragma HLS INTERFACE s_axilite port=length bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATA_PACK variable=in
#pragma HLS DATA_PACK variable=out

    assert(length % PACK_SIZE == 0);
    unsigned size = length / PACK_SIZE;
    PACKED_VAL_T tmp_in;
    PACKED_VAL_T tmp_out;

    loop_kernel_add_scalar_vector_dense:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        tmp_in = in[i];
        for (int k = 0; k < PACK_SIZE; k++) {
            #pragma HLS UNROLL
            tmp_out.data[k] = tmp_in.data[k] + val;
        }
        out[i] = tmp_out;
    }
}

} // extern "C"
