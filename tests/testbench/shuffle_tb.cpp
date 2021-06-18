#include "./shuffle_tb.h"
#include "./shuffle.h"
#include "./util.h"
#include "hls_stream.h"
#include <iostream>
#include <iomanip>

static void data_feeder(
    SF_IO_T input_buffer[NUM_IN_LANES][IN_BUF_SIZE],
    hls::stream<SF_IO_T> input_streams[NUM_IN_LANES],
    hls::stream<unsigned> &num_payloads_in,
    unsigned input_stream_length[NUM_IN_LANES]
) {
    unsigned npld = 0;
    for (unsigned ILid = 0; ILid < NUM_IN_LANES; ILid++) {
        #pragma HLS unroll
        loop_data_feeder:
        for (unsigned i = 0; i < input_stream_length[ILid]; i++) {
            #pragma HLS pipeline II=1
            input_streams[ILid].write(input_buffer[ILid][i]);
            npld ++;
        }
    }
    num_payloads_in.write(npld);
}

static void data_sink(
    hls::stream<SF_IO_T> output_streams[NUM_IN_LANES],
    hls::stream<unsigned> &num_payloads_out,
    SF_IO_T output_buffer[NUM_OUT_LANES][OUT_BUF_SIZE],
    unsigned output_stream_length[NUM_OUT_LANES]
) {
    bool loop_exit = false;
    bool prev_complete = false;

    for (unsigned i = 0; i < NUM_OUT_LANES; i++)  {
        #pragma HLS unroll
        output_stream_length[i] = 0;
    }

    loop_data_sink:
    while (!loop_exit) {
        // #pragma HLS pipeline II=1
        #pragma HLS pipeline off
        bool fifo_allempty = true;
        for (unsigned OLid = 0; OLid < NUM_OUT_LANES; OLid++) {
            #pragma HLS unroll
            SF_IO_T pld;
            bool read_success = output_streams[OLid].read_nb(pld);
            if (read_success) {
                output_buffer[OLid][output_stream_length[OLid]] = pld;
                output_stream_length[OLid]++;
            }
            fifo_allempty = fifo_allempty && !read_success;
        }
        unsigned dummy;
        if (!prev_complete) { prev_complete = num_payloads_out.read_nb(dummy); }
        loop_exit = prev_complete && fifo_allempty;
    }
}

static void main_dataflow(
    SF_IO_T input_buffer[NUM_IN_LANES][IN_BUF_SIZE],
    unsigned input_stream_length[NUM_IN_LANES],
    SF_IO_T output_buffer[NUM_OUT_LANES][OUT_BUF_SIZE],
    unsigned output_stream_length[NUM_OUT_LANES]
) {
    hls::stream<SF_IO_T> input_streams[NUM_IN_LANES];
    #pragma HLS data_pack variable=input_streams
    #pragma HLS array_partition variable=input_streams complete
    #pragma HLS stream variable=input_streams depth=8
    hls::stream<SF_IO_T> output_streams[NUM_IN_LANES];
    #pragma HLS array_partition variable=output_streams complete
    #pragma HLS stream variable=output_streams depth=8

    hls::stream<unsigned> num_payloads_in;
    #pragma HLS stream variable=num_payloads_in depth=2
    hls::stream<unsigned> num_payloads_out;
    #pragma HLS stream variable=num_payloads_out depth=2

    #pragma HLS dataflow

    data_feeder(input_buffer, input_streams, num_payloads_in, input_stream_length);
    #ifndef __SYNTHESIS__
    std::cout << "INFO: [Shuffle TB-device] data feeder complete" << std::endl << std::flush;
    #endif

    shuffler_1p<SF_IO_T, SF_IO_DATA_T, NUM_IN_LANES, NUM_OUT_LANES, ADDR_MASK> (
        input_streams,
        output_streams,
        num_payloads_in,
        num_payloads_out
    );
    #ifndef __SYNTHESIS__
    std::cout << "INFO: [Shuffle TB-device] shuffler-1p complete" << std::endl << std::flush;
    #endif

    data_sink(output_streams, num_payloads_out, output_buffer, output_stream_length);
    #ifndef __SYNTHESIS__
    std::cout << "INFO: [Shuffle TB-device] data sink reset complete" << std::endl << std::flush;
    #endif
}

static void write_back_gmem(
    TB_IFC_T *test_output_stream0,
    TB_IFC_T *test_output_stream1,
    TB_IFC_T *test_output_stream2,
    TB_IFC_T *test_output_stream3,
    TB_IFC_T *test_output_stream4,
    TB_IFC_T *test_output_stream5,
    TB_IFC_T *test_output_stream6,
    TB_IFC_T *test_output_stream7,
    SF_IO_T output_buffer[NUM_OUT_LANES][OUT_BUF_SIZE],
    unsigned output_stream_length[NUM_OUT_LANES],
    unsigned *num_payloads_out
) {
    loop_wb0:
    for (unsigned i = 0; i < output_stream_length[0]; i++) {
        #pragma HLS pipeline II=1
        TB_IFC_T pld;
        pld.uuid = output_buffer[0][i].data.uuid;
        pld.index = output_buffer[0][i].index;
        test_output_stream0[i] = pld;
    }
    loop_wb1:
    for (unsigned i = 0; i < output_stream_length[1]; i++) {
        #pragma HLS pipeline II=1
        TB_IFC_T pld;
        pld.uuid = output_buffer[1][i].data.uuid;
        pld.index = output_buffer[1][i].index;
        test_output_stream1[i] = pld;
    }
    loop_wb2:
    for (unsigned i = 0; i < output_stream_length[2]; i++) {
        #pragma HLS pipeline II=1
        TB_IFC_T pld;
        pld.uuid = output_buffer[2][i].data.uuid;
        pld.index = output_buffer[2][i].index;
        test_output_stream2[i] = pld;
    }
    loop_wb3:
    for (unsigned i = 0; i < output_stream_length[3]; i++) {
        #pragma HLS pipeline II=1
        TB_IFC_T pld;
        pld.uuid = output_buffer[3][i].data.uuid;
        pld.index = output_buffer[3][i].index;
        test_output_stream3[i] = pld;
    }
    loop_wb4:
    for (unsigned i = 0; i < output_stream_length[4]; i++) {
        #pragma HLS pipeline II=1
        TB_IFC_T pld;
        pld.uuid = output_buffer[4][i].data.uuid;
        pld.index = output_buffer[4][i].index;
        test_output_stream4[i] = pld;
    }
    loop_wb5:
    for (unsigned i = 0; i < output_stream_length[5]; i++) {
        #pragma HLS pipeline II=1
        TB_IFC_T pld;
        pld.uuid = output_buffer[5][i].data.uuid;
        pld.index = output_buffer[5][i].index;
        test_output_stream5[i] = pld;
    }
    loop_wb6:
    for (unsigned i = 0; i < output_stream_length[6]; i++) {
        #pragma HLS pipeline II=1
        TB_IFC_T pld;
        pld.uuid = output_buffer[6][i].data.uuid;
        pld.index = output_buffer[6][i].index;
        test_output_stream6[i] = pld;
    }
    loop_wb7:
    for (unsigned i = 0; i < output_stream_length[7]; i++) {
        #pragma HLS pipeline II=1
        TB_IFC_T pld;
        pld.uuid = output_buffer[7][i].data.uuid;
        pld.index = output_buffer[7][i].index;
        test_output_stream7[i] = pld;
    }

    loop_report_output_len:
    for (unsigned OLid = 0; OLid < NUM_OUT_LANES; OLid++) {
        #pragma HLS pipeline II=1
        num_payloads_out[OLid] = output_stream_length[OLid];
    }
}

extern "C" {
void shuffle_tb(
    const TB_IFC_T *test_input_stream0, //0
    const TB_IFC_T *test_input_stream1, //1
    const TB_IFC_T *test_input_stream2, //2
    const TB_IFC_T *test_input_stream3, //3
    const TB_IFC_T *test_input_stream4, //4
    const TB_IFC_T *test_input_stream5, //5
    const TB_IFC_T *test_input_stream6, //6
    const TB_IFC_T *test_input_stream7, //7
    TB_IFC_T *test_output_stream0,    //8
    TB_IFC_T *test_output_stream1,    //9
    TB_IFC_T *test_output_stream2,    //10
    TB_IFC_T *test_output_stream3,    //11
    TB_IFC_T *test_output_stream4,    //12
    TB_IFC_T *test_output_stream5,    //13
    TB_IFC_T *test_output_stream6,    //14
    TB_IFC_T *test_output_stream7,    //15
    unsigned *num_payloads_in,      //16
    unsigned *num_payloads_out      //17
) {
    #pragma HLS interface m_axi port=test_input_stream0 offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=test_input_stream1 offset=slave bundle=gmem1
    #pragma HLS interface m_axi port=test_input_stream2 offset=slave bundle=gmem2
    #pragma HLS interface m_axi port=test_input_stream3 offset=slave bundle=gmem3
    #pragma HLS interface m_axi port=test_input_stream4 offset=slave bundle=gmem4
    #pragma HLS interface m_axi port=test_input_stream5 offset=slave bundle=gmem5
    #pragma HLS interface m_axi port=test_input_stream6 offset=slave bundle=gmem6
    #pragma HLS interface m_axi port=test_input_stream7 offset=slave bundle=gmem7

    #pragma HLS interface m_axi port=test_output_stream0 offset=slave bundle=gmem8
    #pragma HLS interface m_axi port=test_output_stream1 offset=slave bundle=gmem9
    #pragma HLS interface m_axi port=test_output_stream2 offset=slave bundle=gmem10
    #pragma HLS interface m_axi port=test_output_stream3 offset=slave bundle=gmem11
    #pragma HLS interface m_axi port=test_output_stream4 offset=slave bundle=gmem12
    #pragma HLS interface m_axi port=test_output_stream5 offset=slave bundle=gmem13
    #pragma HLS interface m_axi port=test_output_stream6 offset=slave bundle=gmem14
    #pragma HLS interface m_axi port=test_output_stream7 offset=slave bundle=gmem15

    #pragma HLS interface s_axilite port=test_input_stream0 bundle=control
    #pragma HLS interface s_axilite port=test_input_stream1 bundle=control
    #pragma HLS interface s_axilite port=test_input_stream2 bundle=control
    #pragma HLS interface s_axilite port=test_input_stream3 bundle=control
    #pragma HLS interface s_axilite port=test_input_stream4 bundle=control
    #pragma HLS interface s_axilite port=test_input_stream5 bundle=control
    #pragma HLS interface s_axilite port=test_input_stream6 bundle=control
    #pragma HLS interface s_axilite port=test_input_stream7 bundle=control

    #pragma HLS interface s_axilite port=test_output_stream0 bundle=control
    #pragma HLS interface s_axilite port=test_output_stream1 bundle=control
    #pragma HLS interface s_axilite port=test_output_stream2 bundle=control
    #pragma HLS interface s_axilite port=test_output_stream3 bundle=control
    #pragma HLS interface s_axilite port=test_output_stream4 bundle=control
    #pragma HLS interface s_axilite port=test_output_stream5 bundle=control
    #pragma HLS interface s_axilite port=test_output_stream6 bundle=control
    #pragma HLS interface s_axilite port=test_output_stream7 bundle=control

    #pragma HLS interface m_axi port=num_payloads_in offset=slave bundle=gmem16
    #pragma HLS interface m_axi port=num_payloads_out offset=slave bundle=gmem17

    #pragma HLS interface s_axilite port=num_payloads_in bundle=control
    #pragma HLS interface s_axilite port=num_payloads_out bundle=control

    #pragma HLS interface s_axilite port=return bundle=control

    #pragma HLS data_pack variable=test_input_stream0
    #pragma HLS data_pack variable=test_input_stream1
    #pragma HLS data_pack variable=test_input_stream2
    #pragma HLS data_pack variable=test_input_stream3
    #pragma HLS data_pack variable=test_input_stream4
    #pragma HLS data_pack variable=test_input_stream5
    #pragma HLS data_pack variable=test_input_stream6
    #pragma HLS data_pack variable=test_input_stream7

    #pragma HLS data_pack variable=test_output_stream0
    #pragma HLS data_pack variable=test_output_stream1
    #pragma HLS data_pack variable=test_output_stream2
    #pragma HLS data_pack variable=test_output_stream3
    #pragma HLS data_pack variable=test_output_stream4
    #pragma HLS data_pack variable=test_output_stream5
    #pragma HLS data_pack variable=test_output_stream6
    #pragma HLS data_pack variable=test_output_stream7

    // input buffer
    SF_IO_T input_buffer[NUM_IN_LANES][IN_BUF_SIZE];
    #pragma HLS array_partition variable=input_buffer dim=1 complete
    #pragma HLS data_pack variable=input_buffer

    // output buffer
    SF_IO_T output_buffer[NUM_OUT_LANES][OUT_BUF_SIZE];
    #pragma HLS array_partition variable=output_buffer dim=1 complete
    #pragma HLS data_pack variable=output_buffer

    // reset output buffer
    loop_reset_ob:
    for (unsigned i = 0; i < OUT_BUF_SIZE; i++) {
        #pragma HLS pipeline II=1
        for (unsigned OLid = 0; OLid < NUM_OUT_LANES; OLid++) {
            #pragma HLS unroll
            output_buffer[OLid][i] = (SF_IO_T){OLid, (SF_IO_DATA_T){INVALID_UUID, 0}};
        }
    }
    #ifndef __SYNTHESIS__
    std::cout << "INFO: [Shuffle TB-device] output buffer reset complete" << std::endl << std::flush;
    #endif

    // get input stream length
    unsigned input_stream_length[NUM_IN_LANES];
    #pragma HLS array_partition variable=input_stream_length complete
    loop_get_in_stream_len:
    for (unsigned ILid = 0; ILid < NUM_IN_LANES; ILid++) {
        #pragma HLS pipeline II=1
        input_stream_length[ILid] = num_payloads_in[ILid];
    }
    #ifndef __SYNTHESIS__
    std::cout << "INFO: [Shuffle TB-device] input stream length inithalized" << std::endl << std::flush;
    #endif

    // reset output stream length
    unsigned output_stream_length[NUM_OUT_LANES];
    #pragma HLS array_partition variable=output_stream_length complete
    for (unsigned OLid = 0; OLid < NUM_OUT_LANES; OLid++)  {
        #pragma HLS unroll
        output_stream_length[OLid] = 0;
    }
    #ifndef __SYNTHESIS__
    std::cout << "INFO: [Shuffle TB-device] output stream length reset complete" << std::endl << std::flush;
    #endif

    // initialize input buffer
    loop_ini_ib0:
    for (unsigned i = 0; i < input_stream_length[0]; i++) {
        #pragma HLS pipeline II=1
        SF_IO_T pld;
        pld.index = test_input_stream0[i].index;
        pld.data.uuid = test_input_stream0[i].uuid;
        pld.data.padding = 0;
        input_buffer[0][i] = pld;
    }
    loop_ini_ib1:
    for (unsigned i = 0; i < input_stream_length[1]; i++) {
        #pragma HLS pipeline II=1
        SF_IO_T pld;
        pld.index = test_input_stream1[i].index;
        pld.data.uuid = test_input_stream1[i].uuid;
        pld.data.padding = 0;
        input_buffer[1][i] = pld;
    }
    loop_ini_ib2:
    for (unsigned i = 0; i < input_stream_length[2]; i++) {
        #pragma HLS pipeline II=1
        SF_IO_T pld;
        pld.index = test_input_stream2[i].index;
        pld.data.uuid = test_input_stream2[i].uuid;
        pld.data.padding = 0;
        input_buffer[2][i] = pld;
    }
    loop_ini_ib3:
    for (unsigned i = 0; i < input_stream_length[3]; i++) {
        #pragma HLS pipeline II=1
        SF_IO_T pld;
        pld.index = test_input_stream3[i].index;
        pld.data.uuid = test_input_stream3[i].uuid;
        pld.data.padding = 0;
        input_buffer[3][i] = pld;
    }
    loop_ini_ib4:
    for (unsigned i = 0; i < input_stream_length[4]; i++) {
        #pragma HLS pipeline II=1
        SF_IO_T pld;
        pld.index = test_input_stream4[i].index;
        pld.data.uuid = test_input_stream4[i].uuid;
        pld.data.padding = 0;
        input_buffer[4][i] = pld;
    }
    loop_ini_ib5:
    for (unsigned i = 0; i < input_stream_length[5]; i++) {
        #pragma HLS pipeline II=1
        SF_IO_T pld;
        pld.index = test_input_stream5[i].index;
        pld.data.uuid = test_input_stream5[i].uuid;
        pld.data.padding = 0;
        input_buffer[5][i] = pld;
    }
    loop_ini_ib6:
    for (unsigned i = 0; i < input_stream_length[6]; i++) {
        #pragma HLS pipeline II=1
        SF_IO_T pld;
        pld.index = test_input_stream6[i].index;
        pld.data.uuid = test_input_stream6[i].uuid;
        pld.data.padding = 0;
        input_buffer[6][i] = pld;
    }
    loop_ini_ib7:
    for (unsigned i = 0; i < input_stream_length[7]; i++) {
        #pragma HLS pipeline II=1
        SF_IO_T pld;
        pld.index = test_input_stream7[i].index;
        pld.data.uuid = test_input_stream7[i].uuid;
        pld.data.padding = 0;
        input_buffer[7][i] = pld;
    }
    #ifndef __SYNTHESIS__
    std::cout << "INFO: [Shuffle TB-device] input buffer initialized" << std::endl << std::flush;
    #endif

    // run main dataflow
    main_dataflow(input_buffer, input_stream_length, output_buffer, output_stream_length);
    #ifndef __SYNTHESIS__
    std::cout << "INFO: [Shuffle TB-device] mian-dataflow finished" << std::endl << std::flush;
    #endif

    // write back results to gmem
    write_back_gmem(
        test_output_stream0,
        test_output_stream1,
        test_output_stream2,
        test_output_stream3,
        test_output_stream4,
        test_output_stream5,
        test_output_stream6,
        test_output_stream7,
        output_buffer,
        output_stream_length,
        num_payloads_out
    );
    #ifndef __SYNTHESIS__
    std::cout << "INFO: [Shuffle TB-device] kernel finished." << std::endl << std::flush;
    #endif

} // extern "C"
} // kernel
