#ifndef GRAPHLILY_OVERLAY_SYNTHESIZER_H_
#define GRAPHLILY_OVERLAY_SYNTHESIZER_H_

#include "graphlily/synthesizer/base_synthesizer.h"


namespace graphlily {
namespace synthesizer {

class OverlaySynthesizer : public BaseSynthesizer {
private:
    // Kernel configuration
    uint32_t num_channels_;
    uint32_t spmv_out_buf_len_;
    uint32_t spmspv_out_buf_len_;
    uint32_t vec_buf_len_;

public:
    OverlaySynthesizer(uint32_t num_channels,
                       uint32_t spmv_out_buf_len,
                       uint32_t spmspv_out_buf_len,
                       uint32_t vec_buf_len) : BaseSynthesizer("overlay") {
        this->num_channels_ = num_channels;
        this->spmv_out_buf_len_ = spmv_out_buf_len;
        this->spmspv_out_buf_len_ = spmspv_out_buf_len;
        this->vec_buf_len_ = vec_buf_len;
    }

    void generate_kernel_header() override;

    void generate_kernel_ini() override;
};


void OverlaySynthesizer::generate_kernel_header() {
    std::string command = "mkdir -p " + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream header(graphlily::proj_folder_name + "/" + this->kernel_name_ + ".h", std::ios_base::app);
    header << "const unsigned SPMV_OUT_BUF_LEN = " << this->spmv_out_buf_len_ << ";" << std::endl;
    header << "const unsigned SPMSPV_OUT_BUF_LEN = " << this->spmspv_out_buf_len_ << ";" << std::endl;
    header << "const unsigned VEC_BUF_LEN = " << this->vec_buf_len_ << ";" << std::endl;
    header << "#define NUM_HBM_CHANNEL " << this->num_channels_ << std::endl;
    header << "#define SPMV_NUM_PE_TOTAL " << this->num_channels_ * graphlily::pack_size << std::endl;
    header << std::endl;
    header << "#endif  // GRAPHLILY_HW_OVERLAY_H_" << std::endl;
    header.close();
}


void OverlaySynthesizer::generate_kernel_ini() {
    std::string command = "mkdir -p " + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream ini(graphlily::proj_folder_name + "/" + this->kernel_name_ + ".ini");
    ini << "[connectivity]" << std::endl;
    // SpMV
    for (size_t hbm_idx = 0; hbm_idx < this->num_channels_; hbm_idx++) {
        ini << "sp=overlay_1.spmv_channel_" << hbm_idx << "_matrix:HBM["
            << hbm_idx << "]" << std::endl;
    }
    ini << "sp=overlay_1.spmv_vector:HBM[20]" << std::endl;
    ini << "sp=overlay_1.spmv_mask:HBM[21]" << std::endl;
    ini << "sp=overlay_1.spmv_mask_w:HBM[21]" << std::endl;
    ini << "sp=overlay_1.spmv_out:HBM[22]" << std::endl;
    // SpMSpV
    ini << "sp=overlay_1.spmspv_matrix:DDR[0]" << std::endl;
    ini << "sp=overlay_1.spmspv_matrix_indptr:DDR[0]" << std::endl;
    ini << "sp=overlay_1.spmspv_matrix_partptr:DDR[0]" << std::endl;
    ini << "sp=overlay_1.spmspv_vector:HBM[20]" << std::endl;
    ini << "sp=overlay_1.spmspv_mask:HBM[21]" << std::endl;
    ini << "sp=overlay_1.spmspv_out:HBM[22]" << std::endl;
    // enable retiming
    ini << "[vivado]" << std::endl;
    ini << "prop=run.__KERNEL__.{STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS}={-retiming}" << std::endl;
    ini.close();
}


}  // namespace synthesizer
}  // namespace graphlily

#endif  // GRAPHLILY_OVERLAY_SYNTHESIZER_H_
