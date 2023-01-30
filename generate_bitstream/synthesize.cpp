#include "graphlily/synthesizer/split_kernel_synthesizer.h"


int main(int argc, char *argv[]) {
    uint32_t spmv_out_buf_len = 1024 * 1024;
    uint32_t spmspv_out_buf_len = 256 * 1024;
    uint32_t spmv_vec_buf_len = 32 * 1024;
    uint32_t num_hbm_channels = 16;

    graphlily::synthesizer::SplitKernelSynthesizer synthesizer(graphlily::num_hbm_channels,
                                                            spmv_out_buf_len,
                                                            spmspv_out_buf_len,
                                                            spmv_vec_buf_len);
    synthesizer.set_target("hw");
    synthesizer.synthesize();
}
