#include "kernel_strided_access_v1.h"


extern "C" {

void kernel_strided_access_v1(
    const v_data_t *in0,               // Read-only input in HBM channel 0
    v_data_t *out,                     // Out in HBM channel 1
    const unsigned int num_times       // Running the same kernel num_times for performance measurement
) {
#pragma HLS INTERFACE m_axi port=in0 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1

#pragma HLS INTERFACE s_axilite port=in0 bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control

#pragma HLS INTERFACE s_axilite port=num_times bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATA_PACK variable=in0
#pragma HLS DATA_PACK variable=out

    v_data_t tmpIn0;

    // Running the same kernel num_times for performance measurement
    for (int count = 0; count < num_times; count++) {

        loop_x:
        for (int i = 0; i < IN_SIZE / VDATA_SIZE; i+=STRIDE) {
            #pragma HLS PIPELINE II=1
            tmpIn0 = in0[i];
            for (int k = 0; k < VDATA_SIZE; k++) {
                tmpIn0.data[k] += 1;
            }
            out[i] = tmpIn0;
        }
    }

}

} // extern "C"
