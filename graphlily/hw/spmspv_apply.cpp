#include "./libfpga/hisparse.h"

#include "./kernel_spmspv_impl.h"

#include "./kernel_add_scalar_vector_dense_impl.h"
#include "./kernel_assign_vector_dense_impl.h"
#include "./kernel_assign_vector_sparse_no_new_frontier_impl.h"
#include "./kernel_assign_vector_sparse_new_frontier_impl.h"


extern "C" {

void spmspv_apply(
    PACKED_VAL_T *spmv_vector,              // NUM_HBM_CHANNEL + 0
    PACKED_VAL_T *spmv_mask,                // NUM_HBM_CHANNEL + 1
    PACKED_VAL_T *spmv_mask_w,              // NUM_HBM_CHANNEL + 2, used for write into spmv_mask
    PACKED_VAL_T *spmv_out,                 // NUM_HBM_CHANNEL + 3
    /*----------------- arguments for SpMSpV -------------------*/
    const SPMSPV_MAT_PKT_T *spmspv_matrix,  // NUM_HBM_CHANNEL + 4
    const IDX_T *spmspv_matrix_indptr,      // NUM_HBM_CHANNEL + 5
    const IDX_T *spmspv_matrix_partptr,     // NUM_HBM_CHANNEL + 6
    IDX_VAL_T *spmspv_vector,               // NUM_HBM_CHANNEL + 7
    VAL_T *spmspv_mask,                     // NUM_HBM_CHANNEL + 8
    IDX_VAL_T *spmspv_out,                  // NUM_HBM_CHANNEL + 9
    /*-------- arguments shared by all kernels -------------*/
    unsigned num_rows,                      // NUM_HBM_CHANNEL + 10
    unsigned num_cols,                      // NUM_HBM_CHANNEL + 11
    OP_T Op,                                // NUM_HBM_CHANNEL + 12
    MASK_T mask_type,                       // NUM_HBM_CHANNEL + 13
    unsigned mode,                          // NUM_HBM_CHANNEL + 14
    /*-------- arguments for apply kernels -------------*/
    // val must be the last argument, otherwise fixed point causes an XRT run-time error
    unsigned length,                        // NUM_HBM_CHANNEL + 15
    unsigned val_ufixed                     // NUM_HBM_CHANNEL + 16
) {

#pragma HLS INTERFACE m_axi port=spmv_vector offset=slave bundle=spmv_gmem32
#pragma HLS INTERFACE m_axi port=spmv_mask offset=slave bundle=spmv_gmem33
#pragma HLS INTERFACE m_axi port=spmv_mask_w offset=slave bundle=spmv_gmem34
#pragma HLS INTERFACE m_axi port=spmv_out offset=slave bundle=spmv_gmem35

#pragma HLS INTERFACE s_axilite port=spmv_vector bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_mask bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_mask_w bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_out bundle=control

/*----------------- arguments for SpMSpV -------------------*/
#pragma HLS interface m_axi port=spmspv_matrix         offset=slave bundle=spmspv_gmem0
#pragma HLS interface m_axi port=spmspv_matrix_indptr  offset=slave bundle=spmspv_gmem1
#pragma HLS interface m_axi port=spmspv_matrix_partptr offset=slave bundle=spmspv_gmem2
#pragma HLS interface m_axi port=spmspv_vector         offset=slave bundle=spmspv_gmem3
#pragma HLS interface m_axi port=spmspv_mask           offset=slave bundle=spmspv_gmem4
#pragma HLS interface m_axi port=spmspv_out            offset=slave bundle=spmspv_gmem5

#pragma HLS interface s_axilite port=spmspv_matrix         bundle=control
#pragma HLS interface s_axilite port=spmspv_matrix_indptr  bundle=control
#pragma HLS interface s_axilite port=spmspv_matrix_partptr bundle=control
#pragma HLS interface s_axilite port=spmspv_vector         bundle=control
#pragma HLS interface s_axilite port=spmspv_mask           bundle=control
#pragma HLS interface s_axilite port=spmspv_out            bundle=control

/*-------- arguments shared by all kernels ---------*/
#pragma HLS INTERFACE s_axilite port=num_rows bundle=control
#pragma HLS INTERFACE s_axilite port=num_cols bundle=control
#pragma HLS INTERFACE s_axilite port=Op bundle=control
#pragma HLS INTERFACE s_axilite port=mask_type bundle=control
#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

/*-------- arguments for apply kernels -------------*/
#pragma HLS INTERFACE s_axilite port=length bundle=control
#pragma HLS INTERFACE s_axilite port=val_ufixed bundle=control

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

    // VAL_T out_uram_spmspv[NUM_HBM_CHANNEL][PACK_SIZE][SPMSPV_OUT_BUF_LEN / SPMV_NUM_PE_TOTAL];
    // #pragma HLS ARRAY_PARTITION variable=out_uram_spmspv complete dim=1
    // #pragma HLS ARRAY_PARTITION variable=out_uram_spmspv complete dim=2
    // #pragma HLS resource variable=out_uram_spmspv core=RAM_2P latency=3
    // // #pragma HLS resource variable=out_uram_spmspv core=XPM_MEMORY uram latency=2

    /*
        If we do not specify the latency here, the tool will automatically decide the latency of the URAM,
        which could cause problems for the PE due to RAW hazards.
        The URAM latency could be 1, 2, 3, or 4. If specified, it will be applied to both read and write.
    */

    switch (mode) {
        case 2:
            kernel_spmspv(
                spmspv_matrix,
                spmspv_matrix_indptr,
                spmspv_matrix_partptr,
                spmspv_vector,
                spmspv_mask,
                spmspv_out,
                num_rows,
                num_cols,
                Op,
                mask_type
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
