#include "graphlily/synthesizer/overlay_synthesizer.h"


int main(int argc, char *argv[]) {
    uint32_t spmv_out_buf_len = 1000 * 1024;
    uint32_t spmspv_out_buf_len = 250 * 1024;
    uint32_t vec_buf_len = 30 * 1024;
    uint32_t num_hbm_channels = 16;

    graphlily::synthesizer::OverlaySynthesizer synthesizer(num_hbm_channels,
                                                           spmv_out_buf_len,
                                                           spmspv_out_buf_len,
                                                           vec_buf_len);
    synthesizer.set_target("hw");
    synthesizer.synthesize();
}
