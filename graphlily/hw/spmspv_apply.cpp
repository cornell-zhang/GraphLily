#include "./libfpga/hisparse.h"

#include "./kernel_spmspv_impl.h"

#include "./kernel_add_scalar_vector_dense_impl.h"
#include "./kernel_assign_vector_dense_impl.h"
#include "./kernel_assign_vector_sparse_no_new_frontier_impl.h"
#include "./kernel_assign_vector_sparse_new_frontier_impl.h"


extern "C" {

void spmspv_apply(
    /*------------------ arguments for SpMV --------------------*/
    PACKED_VAL_T *spmv_vector,              // inout, HBM[20]
    const PACKED_VAL_T *spmv_mask,          // in,    HBM[21]
    PACKED_VAL_T *spmv_mask_w,              // out,   HBM[21], write into mask
    const PACKED_VAL_T *spmv_out,           // in,    HBM[22]
    /*----------------- arguments for SpMSpV -------------------*/
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
    const SPMSPV_MAT_PKT_T *spmspv_matrix_0,  // in,  HBM[23]
    const IDX_T *spmspv_matrix_indptr_0,      // in,  HBM[23]
    const IDX_T *spmspv_matrix_partptr_0,     // in,  HBM[23]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
    const SPMSPV_MAT_PKT_T *spmspv_matrix_1,  // in,  HBM[24]
    const IDX_T *spmspv_matrix_indptr_1,      // in,  HBM[24]
    const IDX_T *spmspv_matrix_partptr_1,     // in,  HBM[24]
#endif
    IDX_VAL_T *spmspv_vector,               // inout, HBM[20]
    VAL_T *spmspv_mask,                     // inout, HBM[21]
    IDX_VAL_T *spmspv_out,                  // out,   HBM[22]
    /*-------------- arguments shared by kernels ---------------*/
    IDX_T num_rows,                         // in
    IDX_T num_cols,                         // in
    OP_T semiring,                          // in
    MASK_T mask_type,                       // in
    unsigned mode,                          // in
    unsigned length,                        // in
    unsigned val_ufixed                     // in
) {

/*------------------ arguments for SpMV --------------------*/
#pragma HLS INTERFACE m_axi port=spmv_vector offset=slave bundle=spmv_gmem32
#pragma HLS INTERFACE m_axi port=spmv_mask   offset=slave bundle=spmv_gmem33
#pragma HLS INTERFACE m_axi port=spmv_mask_w offset=slave bundle=spmv_gmem34
#pragma HLS INTERFACE m_axi port=spmv_out    offset=slave bundle=spmv_gmem35

#pragma HLS INTERFACE s_axilite port=spmv_vector bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_mask   bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_mask_w bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_out    bundle=control

/*----------------- arguments for SpMSpV -------------------*/
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)

#pragma HLS interface m_axi port=spmspv_matrix_0         offset=slave bundle=spmspv_gmem0_0
#pragma HLS interface m_axi port=spmspv_matrix_indptr_0  offset=slave bundle=spmspv_gmem1_0
#pragma HLS interface m_axi port=spmspv_matrix_partptr_0 offset=slave bundle=spmspv_gmem2_0

#pragma HLS interface s_axilite port=spmspv_matrix_0         bundle=control
#pragma HLS interface s_axilite port=spmspv_matrix_indptr_0  bundle=control
#pragma HLS interface s_axilite port=spmspv_matrix_partptr_0 bundle=control
#endif

#if (SPMSPV_NUM_HBM_CHANNEL >= 2)

#pragma HLS interface m_axi port=spmspv_matrix_1         offset=slave bundle=spmspv_gmem0_1
#pragma HLS interface m_axi port=spmspv_matrix_indptr_1  offset=slave bundle=spmspv_gmem1_1
#pragma HLS interface m_axi port=spmspv_matrix_partptr_1 offset=slave bundle=spmspv_gmem2_1

#pragma HLS interface s_axilite port=spmspv_matrix_1         bundle=control
#pragma HLS interface s_axilite port=spmspv_matrix_indptr_1  bundle=control
#pragma HLS interface s_axilite port=spmspv_matrix_partptr_1 bundle=control

#endif

#pragma HLS interface m_axi port=spmspv_vector         offset=slave bundle=spmspv_gmem3
#pragma HLS interface m_axi port=spmspv_mask           offset=slave bundle=spmspv_gmem4
#pragma HLS interface m_axi port=spmspv_out            offset=slave bundle=spmspv_gmem5

#pragma HLS interface s_axilite port=spmspv_vector         bundle=control
#pragma HLS interface s_axilite port=spmspv_mask           bundle=control
#pragma HLS interface s_axilite port=spmspv_out            bundle=control

/*-------------- arguments shared by kernels ---------------*/
#pragma HLS INTERFACE s_axilite port=num_rows   bundle=control
#pragma HLS INTERFACE s_axilite port=num_cols   bundle=control
#pragma HLS INTERFACE s_axilite port=semiring   bundle=control
#pragma HLS INTERFACE s_axilite port=mask_type  bundle=control
#pragma HLS INTERFACE s_axilite port=mode       bundle=control
#pragma HLS INTERFACE s_axilite port=length     bundle=control
#pragma HLS INTERFACE s_axilite port=val_ufixed bundle=control

#pragma HLS INTERFACE s_axilite port=return bundle=control

// val is used as (1) semiring zero in SpMSpV, (2) input value in apply kernel
VAL_T val;
LOAD_RAW_BITS_FROM_UINT(val, val_ufixed);

#ifndef __SYNTHESIS__
    switch (mode) {
        case 2:
            std::cout << "Running SpMSpV" << std::endl;
            break;
        case 3:
            std::cout << "Running kernel_add_scalar_vector_dense" << std::endl;
            break;
        case 4:
            std::cout << "Running kernel_assign_vector_dense" << std::endl;
            break;
        case 5:
            std::cout << "Running kernel_assign_vector_sparse_no_new_frontier" << std::endl;
            break;
        case 6:
            std::cout << "Running kernel_assign_vector_sparse_new_frontier" << std::endl;
            break;
        default:
            std::cout << "ERROR! Unsupported mode: " << mode << std::endl;
            break;
    }
#endif

    switch (mode) {
        case 2:
            kernel_spmspv(
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
                spmspv_matrix_0,
                spmspv_matrix_indptr_0,
                spmspv_matrix_partptr_0,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
                spmspv_matrix_1,
                spmspv_matrix_indptr_1,
                spmspv_matrix_partptr_1,
#endif
                spmspv_vector,
                spmspv_mask,
                spmspv_out,
                num_rows,
                num_cols,
                semiring,
                mask_type,
                val
            );
            break;
        case 3:
            kernel_add_scalar_vector_dense(
                spmv_out,
                spmv_vector,
                length,
                val
            );
            break;
        case 4:
            kernel_assign_vector_dense(
                spmv_vector,
                spmv_mask,
                spmv_mask_w,
                length,
                val,
                mask_type
            );
            break;
        case 5:
            kernel_assign_vector_sparse_no_new_frontier(
                spmspv_vector,
                spmspv_mask,
                val
            );
            break;
        case 6:
            kernel_assign_vector_sparse_new_frontier(
                spmspv_out,
                spmspv_mask,
                spmspv_vector
            );
            break;
        default:
            break;
    }
}

}  // extern "C"
