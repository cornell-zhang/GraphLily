#include "./overlay.h"

#include "./kernel_spmv_impl.h"
#include "./kernel_spmspv_impl.h"

#include "./kernel_add_scalar_vector_dense_impl.h"
#include "./kernel_assign_vector_dense_impl.h"
#include "./kernel_assign_vector_sparse_no_new_frontier_impl.h"
#include "./kernel_assign_vector_sparse_new_frontier_impl.h"


extern "C" {

void overlay(
    /*----------------- arguments for SpMV --------------------*/
#if (NUM_HBM_CHANNEL >= 1)
    const SPMV_MAT_PKT_T *spmv_channel_0_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 2)
    const SPMV_MAT_PKT_T *spmv_channel_1_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 4)
    const SPMV_MAT_PKT_T *spmv_channel_2_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_3_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 8)
    const SPMV_MAT_PKT_T *spmv_channel_4_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_5_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_6_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_7_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 16)
    const SPMV_MAT_PKT_T *spmv_channel_8_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_9_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_10_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_11_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_12_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_13_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_14_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_15_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 32)
    const SPMV_MAT_PKT_T *spmv_channel_16_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_17_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_18_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_19_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_20_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_21_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_22_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_23_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_24_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_25_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_26_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_27_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_28_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_29_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_30_matrix,
    const SPMV_MAT_PKT_T *spmv_channel_31_matrix,
#endif
    PACKED_VAL_T *spmv_vector,              // NUM_HBM_CHANNEL + 0
    PACKED_VAL_T *spmv_mask,                // NUM_HBM_CHANNEL + 1
    PACKED_VAL_T *spmv_out,                 // NUM_HBM_CHANNEL + 2
    /*----------------- arguments for SpMSpV -------------------*/
    const SPMSPV_MAT_PKT_T *spmspv_matrix,  // NUM_HBM_CHANNEL + 3
    const IDX_T *spmspv_matrix_indptr,      // NUM_HBM_CHANNEL + 4
    const IDX_T *spmspv_matrix_partptr,     // NUM_HBM_CHANNEL + 5
    IDX_VAL_T *spmspv_vector,               // NUM_HBM_CHANNEL + 6
    VAL_T *spmspv_mask,                     // NUM_HBM_CHANNEL + 7
    IDX_VAL_T *spmspv_out,                  // NUM_HBM_CHANNEL + 8
    /*-------- arguments for apply kernels -------------*/
    unsigned length,                        // NUM_HBM_CHANNEL + 9
    VAL_T val,                              // NUM_HBM_CHANNEL + 10
    /*-------- arguments shared by all kernels -------------*/
    unsigned num_rows,                      // NUM_HBM_CHANNEL + 11
    unsigned num_cols,                      // NUM_HBM_CHANNEL + 12
    OP_T Op,                                // NUM_HBM_CHANNEL + 13
    MASK_T mask_type,                       // NUM_HBM_CHANNEL + 14
    unsigned mode                           // NUM_HBM_CHANNEL + 15
) {
/*----------------- arguments for SpMV -------------------*/
#if (NUM_HBM_CHANNEL >= 1)
#pragma HLS INTERFACE m_axi port=spmv_channel_0_matrix offset=slave bundle=spmv_gmem0
#endif
#if (NUM_HBM_CHANNEL >= 2)
#pragma HLS INTERFACE m_axi port=spmv_channel_1_matrix offset=slave bundle=spmv_gmem1
#endif
#if (NUM_HBM_CHANNEL >= 4)
#pragma HLS INTERFACE m_axi port=spmv_channel_2_matrix offset=slave bundle=spmv_gmem2
#pragma HLS INTERFACE m_axi port=spmv_channel_3_matrix offset=slave bundle=spmv_gmem3
#endif
#if (NUM_HBM_CHANNEL >= 8)
#pragma HLS INTERFACE m_axi port=spmv_channel_4_matrix offset=slave bundle=spmv_gmem4
#pragma HLS INTERFACE m_axi port=spmv_channel_5_matrix offset=slave bundle=spmv_gmem5
#pragma HLS INTERFACE m_axi port=spmv_channel_6_matrix offset=slave bundle=spmv_gmem6
#pragma HLS INTERFACE m_axi port=spmv_channel_7_matrix offset=slave bundle=spmv_gmem7
#endif
#if (NUM_HBM_CHANNEL >= 16)
#pragma HLS INTERFACE m_axi port=spmv_channel_8_matrix offset=slave bundle=spmv_gmem8
#pragma HLS INTERFACE m_axi port=spmv_channel_9_matrix offset=slave bundle=spmv_gmem9
#pragma HLS INTERFACE m_axi port=spmv_channel_10_matrix offset=slave bundle=spmv_gmem10
#pragma HLS INTERFACE m_axi port=spmv_channel_11_matrix offset=slave bundle=spmv_gmem11
#pragma HLS INTERFACE m_axi port=spmv_channel_12_matrix offset=slave bundle=spmv_gmem12
#pragma HLS INTERFACE m_axi port=spmv_channel_13_matrix offset=slave bundle=spmv_gmem13
#pragma HLS INTERFACE m_axi port=spmv_channel_14_matrix offset=slave bundle=spmv_gmem14
#pragma HLS INTERFACE m_axi port=spmv_channel_15_matrix offset=slave bundle=spmv_gmem15
#endif
#if (NUM_HBM_CHANNEL >= 32)
#pragma HLS INTERFACE m_axi port=spmv_channel_16_matrix offset=slave bundle=spmv_gmem16
#pragma HLS INTERFACE m_axi port=spmv_channel_17_matrix offset=slave bundle=spmv_gmem17
#pragma HLS INTERFACE m_axi port=spmv_channel_18_matrix offset=slave bundle=spmv_gmem18
#pragma HLS INTERFACE m_axi port=spmv_channel_19_matrix offset=slave bundle=spmv_gmem19
#pragma HLS INTERFACE m_axi port=spmv_channel_20_matrix offset=slave bundle=spmv_gmem20
#pragma HLS INTERFACE m_axi port=spmv_channel_21_matrix offset=slave bundle=spmv_gmem21
#pragma HLS INTERFACE m_axi port=spmv_channel_22_matrix offset=slave bundle=spmv_gmem22
#pragma HLS INTERFACE m_axi port=spmv_channel_23_matrix offset=slave bundle=spmv_gmem23
#pragma HLS INTERFACE m_axi port=spmv_channel_24_matrix offset=slave bundle=spmv_gmem24
#pragma HLS INTERFACE m_axi port=spmv_channel_25_matrix offset=slave bundle=spmv_gmem25
#pragma HLS INTERFACE m_axi port=spmv_channel_26_matrix offset=slave bundle=spmv_gmem26
#pragma HLS INTERFACE m_axi port=spmv_channel_27_matrix offset=slave bundle=spmv_gmem27
#pragma HLS INTERFACE m_axi port=spmv_channel_28_matrix offset=slave bundle=spmv_gmem28
#pragma HLS INTERFACE m_axi port=spmv_channel_29_matrix offset=slave bundle=spmv_gmem29
#pragma HLS INTERFACE m_axi port=spmv_channel_30_matrix offset=slave bundle=spmv_gmem30
#pragma HLS INTERFACE m_axi port=spmv_channel_31_matrix offset=slave bundle=spmv_gmem31
#endif

#pragma HLS INTERFACE m_axi port=spmv_vector offset=slave bundle=spmv_spmspv_gmem0
#pragma HLS INTERFACE m_axi port=spmv_mask offset=slave bundle=spmv_spmspv_gmem1
#pragma HLS INTERFACE m_axi port=spmv_out offset=slave bundle=spmv_spmspv_gmem2

#if (NUM_HBM_CHANNEL >= 1)
#pragma HLS INTERFACE s_axilite port=spmv_channel_0_matrix bundle=control
#endif
#if (NUM_HBM_CHANNEL >= 2)
#pragma HLS INTERFACE s_axilite port=spmv_channel_1_matrix bundle=control
#endif
#if (NUM_HBM_CHANNEL >= 4)
#pragma HLS INTERFACE s_axilite port=spmv_channel_2_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_3_matrix bundle=control
#endif
#if (NUM_HBM_CHANNEL >= 8)
#pragma HLS INTERFACE s_axilite port=spmv_channel_4_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_5_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_6_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_7_matrix bundle=control
#endif
#if (NUM_HBM_CHANNEL >= 16)
#pragma HLS INTERFACE s_axilite port=spmv_channel_8_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_9_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_10_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_11_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_12_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_13_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_14_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_15_matrix bundle=control
#endif
#if (NUM_HBM_CHANNEL >= 32)
#pragma HLS INTERFACE s_axilite port=spmv_channel_16_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_17_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_18_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_19_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_20_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_21_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_22_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_23_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_24_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_25_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_26_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_27_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_28_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_29_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_30_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_channel_31_matrix bundle=control
#endif

#pragma HLS INTERFACE s_axilite port=spmv_vector bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_mask bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_out bundle=control

#if (NUM_HBM_CHANNEL >= 1)
#pragma HLS DATA_PACK variable=spmv_channel_0_matrix
#endif
#if (NUM_HBM_CHANNEL >= 2)
#pragma HLS DATA_PACK variable=spmv_channel_1_matrix
#endif
#if (NUM_HBM_CHANNEL >= 4)
#pragma HLS DATA_PACK variable=spmv_channel_2_matrix
#pragma HLS DATA_PACK variable=spmv_channel_3_matrix
#endif
#if (NUM_HBM_CHANNEL >= 8)
#pragma HLS DATA_PACK variable=spmv_channel_4_matrix
#pragma HLS DATA_PACK variable=spmv_channel_5_matrix
#pragma HLS DATA_PACK variable=spmv_channel_6_matrix
#pragma HLS DATA_PACK variable=spmv_channel_7_matrix
#endif
#if (NUM_HBM_CHANNEL >= 16)
#pragma HLS DATA_PACK variable=spmv_channel_8_matrix
#pragma HLS DATA_PACK variable=spmv_channel_9_matrix
#pragma HLS DATA_PACK variable=spmv_channel_10_matrix
#pragma HLS DATA_PACK variable=spmv_channel_11_matrix
#pragma HLS DATA_PACK variable=spmv_channel_12_matrix
#pragma HLS DATA_PACK variable=spmv_channel_13_matrix
#pragma HLS DATA_PACK variable=spmv_channel_14_matrix
#pragma HLS DATA_PACK variable=spmv_channel_15_matrix
#endif
#if (NUM_HBM_CHANNEL >= 32)
#pragma HLS DATA_PACK variable=spmv_channel_16_matrix
#pragma HLS DATA_PACK variable=spmv_channel_17_matrix
#pragma HLS DATA_PACK variable=spmv_channel_18_matrix
#pragma HLS DATA_PACK variable=spmv_channel_19_matrix
#pragma HLS DATA_PACK variable=spmv_channel_20_matrix
#pragma HLS DATA_PACK variable=spmv_channel_21_matrix
#pragma HLS DATA_PACK variable=spmv_channel_22_matrix
#pragma HLS DATA_PACK variable=spmv_channel_23_matrix
#pragma HLS DATA_PACK variable=spmv_channel_24_matrix
#pragma HLS DATA_PACK variable=spmv_channel_25_matrix
#pragma HLS DATA_PACK variable=spmv_channel_26_matrix
#pragma HLS DATA_PACK variable=spmv_channel_27_matrix
#pragma HLS DATA_PACK variable=spmv_channel_28_matrix
#pragma HLS DATA_PACK variable=spmv_channel_29_matrix
#pragma HLS DATA_PACK variable=spmv_channel_30_matrix
#pragma HLS DATA_PACK variable=spmv_channel_31_matrix
#endif

#pragma HLS DATA_PACK variable=spmv_vector
#pragma HLS DATA_PACK variable=spmv_mask
#pragma HLS DATA_PACK variable=spmv_out

/*----------------- arguments for SpMSpV -------------------*/
#pragma HLS interface m_axi port=spmspv_matrix         offset=slave bundle=spmspv_gmem0
#pragma HLS interface m_axi port=spmspv_matrix_indptr  offset=slave bundle=spmspv_gmem1
#pragma HLS interface m_axi port=spmspv_matrix_partptr offset=slave bundle=spmspv_gmem2
#pragma HLS interface m_axi port=spmspv_vector         offset=slave bundle=spmv_spmspv_gmem0
#pragma HLS interface m_axi port=spmspv_mask           offset=slave bundle=spmv_spmspv_gmem1
#pragma HLS interface m_axi port=spmspv_out            offset=slave bundle=spmv_spmspv_gmem2

#pragma HLS interface s_axilite port=spmspv_matrix         bundle=control
#pragma HLS interface s_axilite port=spmspv_matrix_indptr  bundle=control
#pragma HLS interface s_axilite port=spmspv_matrix_partptr bundle=control
#pragma HLS interface s_axilite port=spmspv_vector         bundle=control
#pragma HLS interface s_axilite port=spmspv_mask           bundle=control
#pragma HLS interface s_axilite port=spmspv_out            bundle=control

#pragma HLS data_pack variable=spmspv_matrix
#pragma HLS data_pack variable=spmspv_vector
#pragma HLS data_pack variable=spmspv_out

/*-------- arguments for apply kernels -------------*/
#pragma HLS INTERFACE s_axilite port=length bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control

/*-------- arguments shared by all kernels ---------*/
#pragma HLS INTERFACE s_axilite port=num_rows bundle=control
#pragma HLS INTERFACE s_axilite port=num_cols bundle=control
#pragma HLS INTERFACE s_axilite port=Op bundle=control
#pragma HLS INTERFACE s_axilite port=mask_type bundle=control
#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    VAL_T out_uram_spmv[NUM_HBM_CHANNEL][PACK_SIZE][OUT_BUF_LEN / SPMV_NUM_PE_TOTAL];
    #pragma HLS ARRAY_PARTITION variable=out_uram_spmv complete dim=1
    #pragma HLS ARRAY_PARTITION variable=out_uram_spmv complete dim=2
    #pragma HLS resource variable=out_uram_spmv core=XPM_MEMORY uram latency=2

    VAL_T out_uram_spmspv[NUM_HBM_CHANNEL][PACK_SIZE][OUT_BUF_LEN / SPMV_NUM_PE_TOTAL];
    #pragma HLS ARRAY_PARTITION variable=out_uram_spmspv complete dim=1
    #pragma HLS ARRAY_PARTITION variable=out_uram_spmspv complete dim=2
    #pragma HLS resource variable=out_uram_spmspv core=XPM_MEMORY uram latency=2

    /*
        If we do not specify the latency here, the tool will automatically decide the latency of the URAM,
        which could cause problems for the PE due to RAW hazards.
        The URAM latency could be 1, 2, 3, or 4. If specified, it will be applied to both read and write.
    */

    switch (mode) {
        case 1: {
            #ifndef __SYNTHESIS__
            std::cout << "Running SpMV" << std::endl;
            #endif
            kernel_spmv(
#if (NUM_HBM_CHANNEL >= 1)
                spmv_channel_0_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 2)
                spmv_channel_1_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 4)
                spmv_channel_2_matrix,
                spmv_channel_3_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 8)
                spmv_channel_4_matrix,
                spmv_channel_5_matrix,
                spmv_channel_6_matrix,
                spmv_channel_7_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 16)
                spmv_channel_8_matrix,
                spmv_channel_9_matrix,
                spmv_channel_10_matrix,
                spmv_channel_11_matrix,
                spmv_channel_12_matrix,
                spmv_channel_13_matrix,
                spmv_channel_14_matrix,
                spmv_channel_15_matrix,
#endif
#if (NUM_HBM_CHANNEL >= 32)
                spmv_channel_16_matrix,
                spmv_channel_17_matrix,
                spmv_channel_18_matrix,
                spmv_channel_19_matrix,
                spmv_channel_20_matrix,
                spmv_channel_21_matrix,
                spmv_channel_22_matrix,
                spmv_channel_23_matrix,
                spmv_channel_24_matrix,
                spmv_channel_25_matrix,
                spmv_channel_26_matrix,
                spmv_channel_27_matrix,
                spmv_channel_28_matrix,
                spmv_channel_29_matrix,
                spmv_channel_30_matrix,
                spmv_channel_31_matrix,
#endif
                spmv_vector,
                spmv_mask,
                spmv_out,
                num_rows,
                num_cols,
                Op,
                mask_type,
                out_uram_spmv
            );
            break;
        }
        case 2: {
            #ifndef __SYNTHESIS__
            std::cout << "Running SpMSpV" << std::endl;
            #endif
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
                mask_type,
                out_uram_spmspv
            );
            break;
        }
        case 3: {
            #ifndef __SYNTHESIS__
            std::cout << "Running kernel_add_scalar_vector_dense" << std::endl;
            #endif
            kernel_add_scalar_vector_dense(spmv_out,
                                           spmv_vector,
                                           length,
                                           val);
            break;
        }
        case 4: {
            #ifndef __SYNTHESIS__
            std::cout << "Running kernel_assign_vector_dense" << std::endl;
            #endif
            kernel_assign_vector_dense(spmv_vector,
                                       spmv_mask,
                                       length,
                                       val,
                                       mask_type);
            break;
        }
        case 5: {
            #ifndef __SYNTHESIS__
            std::cout << "Running kernel_assign_vector_sparse_no_new_frontier" << std::endl;
            #endif
            kernel_assign_vector_sparse_no_new_frontier(spmspv_vector,
                                                        spmspv_mask,
                                                        val);
            break;
        }
        case 6: {
            #ifndef __SYNTHESIS__
            std::cout << "Running kernel_assign_vector_sparse_new_frontier" << std::endl;
            #endif
            kernel_assign_vector_sparse_new_frontier(spmspv_out,
                                                     spmspv_mask,
                                                     spmspv_vector);
            break;
        }
    }
}

}  // extern "C"
