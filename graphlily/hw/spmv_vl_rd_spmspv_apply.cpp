#include "./libfpga/hisparse.h"

#include "./kernel_spmv_vector_loader.h"
#include "./kernel_spmv_result_drain.h"

#include "./kernel_spmspv_impl.h"

#include "./kernel_add_scalar_vector_dense_impl.h"
#include "./kernel_assign_vector_dense_impl.h"
#include "./kernel_assign_vector_sparse_no_new_frontier_impl.h"
#include "./kernel_assign_vector_sparse_new_frontier_impl.h"


extern "C" {

void spmv_vl_rd_spmspv_apply(
    /*------------------ arguments for SpMV --------------------*/
    PACKED_VAL_T *spmv_vector,              // inout, HBM[20]
    const PACKED_VAL_T *spmv_mask,          // in,    HBM[21]
    PACKED_VAL_T *spmv_mask_w,              // out,   HBM[21], write into mask
    PACKED_VAL_T *spmv_out,                 // inout, HBM[22]
    /*----------------- arguments for SpMSpV -------------------*/
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
    SPMSPV_MAT_ARGS(0),                     // in,    HBM[23]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
    SPMSPV_MAT_ARGS(1),                     // in,    HBM[24]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 4)
    SPMSPV_MAT_ARGS(2),                     // in,    HBM[25]
    SPMSPV_MAT_ARGS(3),                     // in,    HBM[26]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 6)
    SPMSPV_MAT_ARGS(4),                     // in,    HBM[27]
    SPMSPV_MAT_ARGS(5),                     // in,    HBM[28]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 8)
    SPMSPV_MAT_ARGS(6),                     // in,    HBM[29]
    SPMSPV_MAT_ARGS(7),                     // in,    HBM[30]
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 10)
    SPMSPV_MAT_ARGS(8),                     // in,    HBM[31]
    SPMSPV_MAT_ARGS(9),                     // in,    HBM[32]
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
    unsigned val_ufixed,                    // in
    unsigned spmv_row_part_id,              // in, only used by SpMV RD so far
    /*-------------- streams for SpMV VL and RD ---------------*/
    hls::stream<VEC_AXIS_T> &to_SLR0,                      // out
    hls::stream<VEC_AXIS_T> &to_SLR1,                      // out
    hls::stream<VEC_AXIS_T> &to_SLR2,
    hls::stream<VEC_AXIS_T> &from_SLR0,     // out
    hls::stream<VEC_AXIS_T> &from_SLR1,     // out
    hls::stream<VEC_AXIS_T> &from_SLR2      // out
) {

/*------------------ arguments for SpMV --------------------*/
#pragma HLS INTERFACE m_axi port=spmv_vector offset=slave bundle=spmv_gmem_vec
#pragma HLS INTERFACE m_axi port=spmv_mask   offset=slave bundle=spmv_gmem_mask
#pragma HLS INTERFACE m_axi port=spmv_mask_w offset=slave bundle=spmv_gmem_mask_w
#pragma HLS INTERFACE m_axi port=spmv_out    offset=slave bundle=spmv_gmem_out

#pragma HLS INTERFACE s_axilite port=spmv_vector bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_mask   bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_mask_w bundle=control
#pragma HLS INTERFACE s_axilite port=spmv_out    bundle=control

#pragma HLS interface axis register both port=to_SLR0
#pragma HLS interface axis register both port=to_SLR1
#pragma HLS interface axis register both port=to_SLR2
#pragma HLS interface axis register both port=from_SLR0
#pragma HLS interface axis register both port=from_SLR1
#pragma HLS interface axis register both port=from_SLR2

/*----------------- arguments for SpMSpV -------------------*/
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
    #pragma HLS interface m_axi     port=spmspv_mat_0 offset=slave bundle=spmspv_gmem0_0 max_read_burst_length=128 max_write_burst_length=2 num_write_outstanding=1
    #pragma HLS interface s_axilite port=spmspv_mat_0 bundle=control
#endif

#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
    #pragma HLS interface m_axi     port=spmspv_mat_1 offset=slave bundle=spmspv_gmem0_1 max_read_burst_length=128 max_write_burst_length=2 num_write_outstanding=1
    #pragma HLS interface s_axilite port=spmspv_mat_1 bundle=control
#endif

#if (SPMSPV_NUM_HBM_CHANNEL >= 4)
    #pragma HLS interface m_axi     port=spmspv_mat_2 offset=slave bundle=spmspv_gmem0_2 max_read_burst_length=128 max_write_burst_length=2 num_write_outstanding=1
    #pragma HLS interface m_axi     port=spmspv_mat_3 offset=slave bundle=spmspv_gmem0_3 max_read_burst_length=128 max_write_burst_length=2 num_write_outstanding=1
    #pragma HLS interface s_axilite port=spmspv_mat_2 bundle=control
    #pragma HLS interface s_axilite port=spmspv_mat_3 bundle=control
#endif

#if (SPMSPV_NUM_HBM_CHANNEL >= 6)
    #pragma HLS interface m_axi     port=spmspv_mat_4 offset=slave bundle=spmspv_gmem0_4 max_read_burst_length=128 max_write_burst_length=2 num_write_outstanding=1
    #pragma HLS interface m_axi     port=spmspv_mat_5 offset=slave bundle=spmspv_gmem0_5 max_read_burst_length=128 max_write_burst_length=2 num_write_outstanding=1
    #pragma HLS interface s_axilite port=spmspv_mat_4 bundle=control
    #pragma HLS interface s_axilite port=spmspv_mat_5 bundle=control
#endif

#if (SPMSPV_NUM_HBM_CHANNEL >= 8)
    #pragma HLS interface m_axi     port=spmspv_mat_6 offset=slave bundle=spmspv_gmem0_6 max_read_burst_length=128 max_write_burst_length=2 num_write_outstanding=1
    #pragma HLS interface m_axi     port=spmspv_mat_7 offset=slave bundle=spmspv_gmem0_7 max_read_burst_length=128 max_write_burst_length=2 num_write_outstanding=1
    #pragma HLS interface s_axilite port=spmspv_mat_6 bundle=control
    #pragma HLS interface s_axilite port=spmspv_mat_7 bundle=control
#endif

#if (SPMSPV_NUM_HBM_CHANNEL >= 10)
    #pragma HLS interface m_axi     port=spmspv_mat_8 offset=slave bundle=spmspv_gmem0_8
    #pragma HLS interface m_axi     port=spmspv_mat_9 offset=slave bundle=spmspv_gmem0_9
    #pragma HLS interface s_axilite port=spmspv_mat_8 bundle=control
    #pragma HLS interface s_axilite port=spmspv_mat_9 bundle=control
#endif

#pragma HLS interface m_axi port=spmspv_vector  offset=slave bundle=spmspv_gmem_vec max_read_burst_length=256 max_write_burst_length=2 num_write_outstanding=1
#pragma HLS interface m_axi port=spmspv_mask    offset=slave bundle=spmspv_gmem4
#pragma HLS interface m_axi port=spmspv_out     offset=slave bundle=spmspv_gmem_out max_read_burst_length=2 num_read_outstanding=1 max_write_burst_length=256 num_write_outstanding=12

#pragma HLS interface s_axilite port=spmspv_vector  bundle=control
#pragma HLS interface s_axilite port=spmspv_mask    bundle=control
#pragma HLS interface s_axilite port=spmspv_out     bundle=control

/*-------------- arguments shared by kernels ---------------*/
#pragma HLS INTERFACE s_axilite port=num_rows         bundle=control
#pragma HLS INTERFACE s_axilite port=num_cols         bundle=control
#pragma HLS INTERFACE s_axilite port=semiring         bundle=control
#pragma HLS INTERFACE s_axilite port=mask_type        bundle=control
#pragma HLS INTERFACE s_axilite port=mode             bundle=control
#pragma HLS INTERFACE s_axilite port=length           bundle=control
#pragma HLS INTERFACE s_axilite port=val_ufixed       bundle=control
#pragma HLS interface s_axilite port=spmv_row_part_id bundle=control

#pragma HLS INTERFACE s_axilite port=return bundle=control

// val is used as (1) semiring zero in SpMSpV, (2) input value in apply kernel
VAL_T val;
LOAD_RAW_BITS_FROM_UINT(val, val_ufixed);

#ifndef __SYNTHESIS__
    switch (mode) {
        case 1:
            std::cout << "Running SpMV Vector Loader and Result Drain" << std::endl;
            break;
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
        case 1:
            kernel_spmv_vector_loader(
                spmv_vector,
                num_cols,
                to_SLR0,
                to_SLR1,
                to_SLR2
            );
            kernel_spmv_result_drain(
                spmv_out,
                spmv_mask,
                spmv_row_part_id,
                val,
                mask_type,
                from_SLR0,
                from_SLR1,
                from_SLR2
            );
        case 2:
            kernel_spmspv(
#if (SPMSPV_NUM_HBM_CHANNEL >= 1)
                spmspv_mat_0,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 2)
                spmspv_mat_1,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 4)
                spmspv_mat_2,
                spmspv_mat_3,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 6)
                spmspv_mat_4,
                spmspv_mat_5,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 8)
                spmspv_mat_6,
                spmspv_mat_7,
#endif
#if (SPMSPV_NUM_HBM_CHANNEL >= 10)
                spmspv_mat_8,
                spmspv_mat_9,
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
