#include "pe_tb.h"
#include "ufixed_pe_fwd.h"
#include "hls_stream.h"
#include <iostream>
#include <iomanip>

template<unsigned num_lanes>
static void data_feeder(
    PE_I_T input_buffer[num_lanes][IN_BUF_SIZE],
    hls::stream<PE_I_T> output_stream[num_lanes],
    hls::stream<IDX_T> &output_npld_stream
) {
    loop_data_feeder:
    for (unsigned i = 0; i < IN_BUF_SIZE; i++) {
        #pragma HLS pipeline II=1
        for (unsigned Lid = 0; Lid < num_lanes; Lid++)  {
            #pragma HLS unroll
            output_stream[Lid].write(input_buffer[Lid][i]);
        }
    }
    output_npld_stream.write(IN_BUF_SIZE * num_lanes);
}


static void main_dataflow(
    PE_I_T input_buffer[NUM_PE][IN_BUF_SIZE],
    VAL_T output_buffer[NUM_PE][BANK_SIZE]
) {
    hls::stream<PE_I_T> DF_to_PE_stream[NUM_PE];
    hls::stream<IDX_T> DF_to_PE_npld_stream;
    #pragma HLS stream variable=DF_to_PE_stream depth=8
    #pragma HLS stream variable=DF_to_PE_npld_stream depth=2

    #pragma HLS dataflow

    data_feeder<NUM_PE>(input_buffer, DF_to_PE_stream, DF_to_PE_npld_stream);

    ufixed_pe_cluster_uram<VAL_T, char, PE_I_T, NUM_PE, BANK_ID_NBITS, BANK_SIZE>(
        DF_to_PE_stream,
        output_buffer,
        MULADD,
        0,
        DF_to_PE_npld_stream
    );
}

extern "C" {
void pe_tb(
    const IDX_T *test_addr_gmem, //0
    const VAL_T *test_mat_gmem,  //1
    const VAL_T *test_vec_gmem,  //2
    VAL_T *result_gmem           //3
) {
    #pragma HLS interface m_axi port=test_addr_gmem offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=test_mat_gmem  offset=slave bundle=gmem1
    #pragma HLS interface m_axi port=test_vec_gmem  offset=slave bundle=gmem2
    #pragma HLS interface m_axi port=result_gmem    offset=slave bundle=gmem3

    #pragma HLS interface s_axilite port=test_addr_gmem bundle=control
    #pragma HLS interface s_axilite port=test_mat_gmem  bundle=control
    #pragma HLS interface s_axilite port=test_vec_gmem  bundle=control
    #pragma HLS interface s_axilite port=result_gmem    bundle=control

    #pragma HLS interface s_axilite port=return bundle=control

    // input buffer
    PE_I_T input_buffer[NUM_PE][IN_BUF_SIZE];
    #pragma HLS array_partition variable=input_buffer dim=1 complete
    #pragma HLS resource variable=input_buffer core=RAM_1P

    // output buffer
    VAL_T output_buffer[NUM_PE][BANK_SIZE];
    #pragma HLS array_partition variable=output_buffer dim=1 complete
    #pragma HLS resource variable=output_buffer core=RAM_2P latency=2

    // reset output buffer
    loop_reset_ob:
    for (unsigned i = 0; i < BANK_SIZE; i++) {
        #pragma HLS pipeline II=1
        for (unsigned PEid = 0; PEid < NUM_PE; PEid++) {
            #pragma HLS unroll
            output_buffer[PEid][i] = 0;
        }
    }

    // initialize input buffer
    loop_ini_ib:
    for (unsigned i = 0; i < NUM_PE * IN_BUF_SIZE; i++) {
        #pragma HLS pipeline II=1
        input_buffer[i / IN_BUF_SIZE][i % IN_BUF_SIZE].index = test_addr_gmem[i];
        input_buffer[i / IN_BUF_SIZE][i % IN_BUF_SIZE].data.mat_val = test_mat_gmem[i];
        input_buffer[i / IN_BUF_SIZE][i % IN_BUF_SIZE].data.vec_val = test_vec_gmem[i];
    }

    // run main dataflow
    main_dataflow(input_buffer, output_buffer);

    // write back to results
    loop_wb_2:
    for (unsigned i = 0; i < BANK_SIZE; i++) {
        loop_wb_1:
        for (unsigned PEid = 0; PEid < NUM_PE; PEid++) {
            #pragma HLS pipeline II=1
            result_gmem[i * NUM_PE + PEid] = output_buffer[PEid][i];
        }
    }

} // extern "C"
} // kernel
