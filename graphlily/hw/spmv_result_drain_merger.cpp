#include <hls_stream.h>
#include <ap_int.h>

#include "libfpga/hisparse.h"

#ifndef __SYNTHESIS__
#include <iostream>
#include <cstdlib>
// #define SPMV_RESULT_DRAIN_LINE_TRACING
#endif

extern "C" {
void spmv_result_drain_merge(
    PACKED_VAL_T *packed_dense_result,      // out
    const PACKED_VAL_T *packed_dense_mask,  // in
    const unsigned row_part_id,             // in
    // const unsigned rows_per_c_in_partition, // in
    const unsigned zero_ufixed,             // in
    const MASK_T mask_type,                 // in
    hls::stream<VEC_AXIS_T> &from_SLR0,     // out
    hls::stream<VEC_AXIS_T> &from_SLR1,     // out
    hls::stream<VEC_AXIS_T> &from_SLR2      // out
) {
    #pragma HLS interface m_axi port=packed_dense_result offset=slave bundle=spmv_vin
    #pragma HLS interface m_axi port=packed_dense_mask offset=slave bundle=spmv_mask
    #pragma HLS interface s_axilite port=packed_dense_result bundle=control
    #pragma HLS interface s_axilite port=packed_dense_mask bundle=control
    #pragma HLS interface s_axilite port=row_part_id bundle=control
    #pragma HLS interface s_axilite port=zero_ufixed bundle=control
    #pragma HLS interface s_axilite port=mask_type bundle=control
    #pragma HLS interface s_axilite port=return bundle=control

    #pragma HLS interface axis register both port=from_SLR0
    #pragma HLS interface axis register both port=from_SLR1
    #pragma HLS interface axis register both port=from_SLR2

    

} // kernel
} // extern "C"
