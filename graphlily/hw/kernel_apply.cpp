#include "./kernel_apply.h"

#include "./kernel_add_scalar_vector_dense_impl.h"
#include "./kernel_assign_vector_dense_impl.h"
#include "./kernel_assign_vector_sparse_no_new_frontier_impl.h"
#include "./kernel_assign_vector_sparse_new_frontier_impl.h"


extern "C" {

void kernel_apply(
    const PACKED_VAL_T *packed_dense_vector_1,
    PACKED_VAL_T *packed_dense_vector_2,
    VAL_T *dense_vector_1,
    const IDX_VAL_T *sparse_vector_1,
    IDX_VAL_T *sparse_vector_2,
    unsigned length,
    VAL_T val,
    MASK_T mask_type,
    unsigned mode
) {
#pragma HLS INTERFACE m_axi port=packed_dense_vector_1 offset=slave bundle=apply_gmem0
#pragma HLS INTERFACE m_axi port=packed_dense_vector_2 offset=slave bundle=apply_gmem1
#pragma HLS INTERFACE m_axi port=dense_vector_1 offset=slave bundle=apply_gmem2
#pragma HLS INTERFACE m_axi port=sparse_vector_1 offset=slave bundle=apply_gmem3
#pragma HLS INTERFACE m_axi port=sparse_vector_2 offset=slave bundle=apply_gmem4

#pragma HLS INTERFACE s_axilite port=packed_dense_vector_1 bundle=control
#pragma HLS INTERFACE s_axilite port=packed_dense_vector_2 bundle=control
#pragma HLS INTERFACE s_axilite port=dense_vector_1 bundle=control
#pragma HLS INTERFACE s_axilite port=sparse_vector_1 bundle=control
#pragma HLS INTERFACE s_axilite port=sparse_vector_2 bundle=control

#pragma HLS INTERFACE s_axilite port=length bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=mask_type bundle=control
#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATA_PACK variable=packed_dense_vector_1
#pragma HLS DATA_PACK variable=packed_dense_vector_2
#pragma HLS DATA_PACK variable=sparse_vector_1
#pragma HLS DATA_PACK variable=sparse_vector_2

    switch (mode) {
        case 1: {
            kernel_add_scalar_vector_dense(packed_dense_vector_1,
                                           packed_dense_vector_2,
                                           length,
                                           val);
            break;
        }
        case 2: {
            kernel_assign_vector_dense(packed_dense_vector_1,
                                       packed_dense_vector_2,
                                       length,
                                       val,
                                       mask_type);
            break;
        }
        case 3: {
            kernel_assign_vector_sparse_no_new_frontier(sparse_vector_1,
                                                        dense_vector_1,
                                                        val);
            break;
        }
        case 4: {
            kernel_assign_vector_sparse_new_frontier(sparse_vector_1,
                                                     dense_vector_1,
                                                     sparse_vector_2);
            break;
        }
    }
}

}  // extern "C"
