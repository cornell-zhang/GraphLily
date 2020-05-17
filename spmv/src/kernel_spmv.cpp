/*******************************************************************************
Description:
    This is the design of SpMV.

*******************************************************************************/

#include "kernel_spmv.h"

extern "C" {

void kernel_spmv(
    const data_t *vector_ddr,           // The dense vector stored in DDR
    const data_t *indptr_ddr,           // Read-only indptr stored in DDR
    const v_data_t *indices_hbm_0,      // Read-only indices stored in HBM channel 0
    const v_data_t *vals_hbm_1,         // Read-only vals stored in HBM channel 1
    const v_data_t *indices_hbm_2,      // Read-only indices stored in HBM channel 2
    const v_data_t *vals_hbm_3,         // Read-only vals stored in HBM channel 3
    data_t *out_ddr,                    // Output of the SpMV kernel stored in DDR
    const unsigned int num_times        // Running the same kernel num_times for performance measurement
) {
#pragma HLS INTERFACE m_axi port=indices_hbm_0 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=vals_hbm_1    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=indices_hbm_2 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=vals_hbm_3    offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=vector_ddr    offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=indptr_ddr    offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=out_ddr       offset=slave bundle=gmem4

#pragma HLS INTERFACE s_axilite port=indices_hbm_0 bundle=control
#pragma HLS INTERFACE s_axilite port=vals_hbm_1    bundle=control
#pragma HLS INTERFACE s_axilite port=indices_hbm_2 bundle=control
#pragma HLS INTERFACE s_axilite port=vals_hbm_3    bundle=control
#pragma HLS INTERFACE s_axilite port=vector_ddr    bundle=control
#pragma HLS INTERFACE s_axilite port=indptr_ddr    bundle=control
#pragma HLS INTERFACE s_axilite port=out_ddr       bundle=control

#pragma HLS INTERFACE s_axilite port=num_times bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATA_PACK variable=indices_hbm_0
#pragma HLS DATA_PACK variable=vals_hbm_1
#pragma HLS DATA_PACK variable=indices_hbm_2
#pragma HLS DATA_PACK variable=vals_hbm_3

    data_t vector_bram[NUM_PE_TOTAL][NUM_COLS];
    #pragma HLS ARRAY_PARTITION variable=vector_bram complete dim=1
    loop_load_vector:
    for (int j = 0; j < NUM_PE_TOTAL; j++) {
        #pragma HLS UNROLL
        for (int i = 0; i < NUM_COLS; i++) {
            #pragma HLS PIPELINE II=1
            vector_bram[j][i] = vector_ddr[i];
        }
    }

    data_t indptr_bram[NUM_ROWS + 1];
    loop_load_indptr:
    for (int i = 0; i < NUM_ROWS + 1; i++) {
        #pragma HLS PIPELINE II=1
        indptr_bram[i] = indptr_ddr[i];
    }

    data_t out_bram[NUM_ROWS];
    #pragma HLS ARRAY_PARTITION variable=out_bram cyclic factor=NUM_PE_TOTAL

    data_t tmp_out[NUM_PE_TOTAL];
    #pragma HLS ARRAY_PARTITION variable=tmp_out complete

    // Running the same kernel num_times for performance measurement
    for (int count = 0; count < num_times; count++) {

        loop_row_iterate:
        for (int row_idx = 0; row_idx < NUM_ROWS; row_idx+=NUM_PE_TOTAL) {

            // Reset tmp_out to 0
            loop_reset_tmp_out:
            for (int PE_id = 0; PE_id < NUM_PE_TOTAL; PE_id++) {
                #pragma HLS UNROLL
                #pragma HLS LOOP_TRIPCOUNT min=NUM_PE_TOTAL max=NUM_PE_TOTAL
                tmp_out[PE_id] = 0;
            }

            int start = indptr_bram[row_idx/NUM_PE_TOTAL];
            int end = indptr_bram[row_idx/NUM_PE_TOTAL + 1];

            loop_dot_product:
            for (int i = 0; i < end - start; i++) {
                #pragma HLS PIPELINE II=1
                v_data_t tmp_indices_0 = indices_hbm_0[i + start];
                v_data_t tmp_vals_0 = vals_hbm_1[i + start];
                v_data_t tmp_indices_1 = indices_hbm_2[i + start];
                v_data_t tmp_vals_1 = vals_hbm_3[i + start];

                for (int PE_id = 0; PE_id < NUM_PE_TOTAL; PE_id++) {
                    #pragma HLS UNROLL
                    #pragma HLS LOOP_TRIPCOUNT min=NUM_PE_TOTAL max=NUM_PE_TOTAL

                    if (PE_id < NUM_PE_PER_HBM_CHANNEL) {
                        tmp_out[PE_id] += tmp_vals_0.data[PE_id]
                                          * vector_bram[PE_id][tmp_indices_0.data[PE_id]];
                    } else {
                        tmp_out[PE_id] += tmp_vals_1.data[PE_id - NUM_PE_PER_HBM_CHANNEL]
                                          * vector_bram[PE_id][tmp_indices_1.data[PE_id - NUM_PE_PER_HBM_CHANNEL]];
                    }
                }
            }

            loop_write_to_out_bram:
            for (int PE_id = 0; PE_id < NUM_PE_TOTAL; PE_id++) {
                #pragma HLS UNROLL
                #pragma HLS LOOP_TRIPCOUNT min=NUM_PE_TOTAL max=NUM_PE_TOTAL
                out_bram[row_idx + PE_id] = tmp_out[PE_id];
            }
        }
    }

    // Copy result back to global memory (DDR)
    loop_write_to_out_ddr:
    for (int i = 0; i < NUM_ROWS; i++) {
        #pragma HLS PIPELINE II=1
        out_ddr[i] = out_bram[i];
    }
}

} // extern "C"
