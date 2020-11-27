#ifndef GRAPHLILY_SPMV_SPMSPV_SYNTHESIZER_H_
#define GRAPHLILY_SPMV_SPMSPV_SYNTHESIZER_H_

#include "graphlily/synthesizer/base_synthesizer.h"


namespace graphlily {
namespace synthesizer {

class SpmvSpmspvSynthesizer : public BaseSynthesizer {
private:
    // Kernel configuration
    uint32_t num_channels_;
    uint32_t out_buf_len_;
    uint32_t vec_buf_len_;

public:
    SpmvSpmspvSynthesizer(uint32_t num_channels,
                          uint32_t out_buf_len,
                          uint32_t vec_buf_len) : BaseSynthesizer("kernel_spmv_spmspv") {
        this->num_channels_ = num_channels;
        this->out_buf_len_ = out_buf_len;
        this->vec_buf_len_ = vec_buf_len;
    }

    void generate_kernel_header() override;

    void generate_kernel_ini() override;
};


void SpmvSpmspvSynthesizer::generate_kernel_header() {
    std::string command = "mkdir -p " + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream header(graphlily::proj_folder_name + "/" + this->kernel_name_ + ".h", std::ios_base::app);
    header << "const unsigned OUT_BUF_LEN = " << this->out_buf_len_ << ";" << std::endl;
    header << "const unsigned VEC_BUF_LEN = " << this->vec_buf_len_ << ";" << std::endl;
    header << "const unsigned NUM_HBM_CHANNEL = " << this->num_channels_ << ";" << std::endl;
    header << "#define NUM_HBM_CHANNEL " << this->num_channels_ << std::endl;
    header << "const unsigned SPMV_NUM_PE_TOTAL = "
           << this->num_channels_ * graphlily::pack_size << ";" << std::endl;
    header << std::endl;
    header << "#endif  // GRAPHLILY_HW_SPMV_SPMSPV_H_" << std::endl;
    header.close();
}


void SpmvSpmspvSynthesizer::generate_kernel_ini() {
    std::string command = "mkdir -p " + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream ini(graphlily::proj_folder_name + "/" + this->kernel_name_ + ".ini");
    ini << "[connectivity]" << std::endl;
    // SpMV
    for (size_t hbm_idx = 0; hbm_idx < this->num_channels_; hbm_idx++) {
        ini << "sp=kernel_spmv_spmspv_1.spmv_channel_" << hbm_idx << "_matrix:HBM["
            << hbm_idx << "]" << std::endl;
    }
    ini << "sp=kernel_spmv_spmspv_1.spmv_vector:DDR[0]" << std::endl;
    ini << "sp=kernel_spmv_spmspv_1.spmv_mask:DDR[0]" << std::endl;
    ini << "sp=kernel_spmv_spmspv_1.spmv_out:DDR[0]" << std::endl;
    // SpMSpV
    ini << "sp=kernel_spmv_spmspv_1.spmspv_vector:DDR[0]" << std::endl;
    ini << "sp=kernel_spmv_spmspv_1.spmspv_mask:DDR[0]" << std::endl;
    ini << "sp=kernel_spmv_spmspv_1.spmspv_out:DDR[0]" << std::endl;
    ini << "sp=kernel_spmv_spmspv_1.spmspv_matrix:DDR[1]" << std::endl;
    ini << "sp=kernel_spmv_spmspv_1.spmspv_matrix_indptr:DDR[1]" << std::endl;
    ini << "sp=kernel_spmv_spmspv_1.spmspv_matrix_partptr:DDR[1]" << std::endl;
    // enable retiming
    ini << "[vivado]" << std::endl;
    ini << "prop=run.__KERNEL__.{STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS}={-retiming}" << std::endl;
    ini.close();
}


}  // namespace synthesizer
}  // namespace graphlily

#endif  // GRAPHLILY_SPMV_SPMSPV_SYNTHESIZER_H_
