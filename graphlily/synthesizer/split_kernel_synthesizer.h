#ifndef GRAPHLILY_OVERLAY_SYNTHESIZER_H_
#define GRAPHLILY_OVERLAY_SYNTHESIZER_H_

#include "graphlily/synthesizer/base_synthesizer.h"


namespace graphlily {
namespace synthesizer {

class SplitKernelSynthesizer : public BaseSynthesizer {
private:
    // Kernel configuration
    uint32_t num_channels_;
    uint32_t spmv_out_buf_len_;
    uint32_t spmspv_out_buf_len_;
    uint32_t spmv_vec_buf_len_;

public:
    SplitKernelSynthesizer(uint32_t num_channels,
                       uint32_t spmv_out_buf_len,
                       uint32_t spmspv_out_buf_len,
                       uint32_t spmv_vec_buf_len_) : BaseSynthesizer("") {
        this->kernel_name_ = "overlay";
        this->num_channels_ = num_channels;
        this->spmv_out_buf_len_ = spmv_out_buf_len;
        this->spmspv_out_buf_len_ = spmspv_out_buf_len;
        this->spmv_vec_buf_len_ = spmv_vec_buf_len_;
        this->makefile_body_ += graphlily::add_kernel_to_makefile("spmv_sk0");
        this->makefile_body_ += graphlily::add_kernel_to_makefile("spmv_sk1");
        this->makefile_body_ += graphlily::add_kernel_to_makefile("spmv_sk2");
        this->makefile_body_ += graphlily::add_kernel_to_makefile("spmv_result_drain");
        this->makefile_body_ += graphlily::add_kernel_to_makefile("spmv_vector_loader");
        this->makefile_body_ += graphlily::add_kernel_to_makefile("k2k_relay");
        this->makefile_body_ += graphlily::add_kernel_to_makefile("spmspv_apply");
    }

    void generate_kernel_header() override;

    void generate_kernel_ini() override;

    void link_kernel_code() override;
};

void SplitKernelSynthesizer::link_kernel_code() {
    std::string command = "cp " + graphlily::root_path + "/graphlily/hw/" + "*.h"
                                + " " + graphlily::proj_folder_name + "/";
    std::cout << command << std::endl;
    system(command.c_str());

    command = "cp " + graphlily::root_path + "/graphlily/hw/" + "k2k_relay.cpp"
                    + " " + graphlily::proj_folder_name + "/" + "k2k_relay.cpp";
    std::cout << command << std::endl;
    system(command.c_str());

    command = "cp " + graphlily::root_path + "/graphlily/hw/" + "spmspv_apply.cpp"
                    + " " + graphlily::proj_folder_name + "/" + "spmspv_apply.cpp";
    std::cout << command << std::endl;
    system(command.c_str());

    command = "cp " + graphlily::root_path + "/graphlily/hw/" + "spmv_result_drain.cpp"
                    + " " + graphlily::proj_folder_name + "/" + "spmv_result_drain.cpp";
    std::cout << command << std::endl;
    system(command.c_str());

    command = "cp " + graphlily::root_path + "/graphlily/hw/" + "spmv_sk0.cpp"
                    + " " + graphlily::proj_folder_name + "/" + "spmv_sk0.cpp";
    std::cout << command << std::endl;
    system(command.c_str());

    command = "cp " + graphlily::root_path + "/graphlily/hw/" + "spmv_sk1.cpp"
                    + " " + graphlily::proj_folder_name + "/" + "spmv_sk1.cpp";
    std::cout << command << std::endl;
    system(command.c_str());

    command = "cp " + graphlily::root_path + "/graphlily/hw/" + "spmv_sk2.cpp"
                    + " " + graphlily::proj_folder_name + "/" + "spmv_sk2.cpp";
    std::cout << command << std::endl;
    system(command.c_str());

    command = "cp " + graphlily::root_path + "/graphlily/hw/" + "spmv_vector_loader.cpp"
                    + " " + graphlily::proj_folder_name + "/" + "spmv_vector_loader.cpp";
    std::cout << command << std::endl;
    system(command.c_str());

    command = "cp -r " + graphlily::root_path + "/graphlily/hw/libfpga"
                    + " " + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());

}


void SplitKernelSynthesizer::generate_kernel_header() {
    std::string command = "mkdir -p " + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream header(graphlily::proj_folder_name + "/libfpga/config.h", std::ios_base::out);
    header << "#ifndef GRAPHLILY_HW_CONFIG_H_" << std::endl;
    header << "#define GRAPHLILY_HW_CONFIG_H_" << std::endl;
    header << "const unsigned SPMV_OUT_BUF_LEN = " << this->spmv_out_buf_len_ << ";" << std::endl;
    header << "const unsigned SPMSPV_OUT_BUF_LEN = " << this->spmspv_out_buf_len_ << ";" << std::endl;
    header << "const unsigned SPMV_VEC_BUF_LEN = " << this->spmv_vec_buf_len_ << ";" << std::endl;
    header << "#define NUM_HBM_CHANNEL " << this->num_channels_ << std::endl;
    header << std::endl;
    header << "#endif  // GRAPHLILY_HW_CONFIG_H_" << std::endl;
    header.close();
}


void SplitKernelSynthesizer::generate_kernel_ini() {
    std::string command = "mkdir -p " + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream ini(graphlily::proj_folder_name + "/" + this->kernel_name_ + ".ini");
    ini << "[connectivity]" << std::endl;

    // SpMV nk tags
    ini << "nk=spmv_vector_loader:1:VL" << std::endl;
    ini << "nk=spmv_result_drain:1:RD" << std::endl;
    ini << "nk=k2k_relay:2:relay_SK2_vin.relay_SK2_rout" << std::endl;

    // SpMV slr tags
    ini << "slr=spmv_sk0_1:SLR0" << std::endl;
    ini << "slr=spmv_sk1_1:SLR1" << std::endl;
    ini << "slr=spmv_sk2_1:SLR2" << std::endl;
    ini << "slr=VL:SLR0" << std::endl;
    ini << "slr=RD:SLR0" << std::endl;
    ini << "slr=relay_SK2_vin:SLR1" << std::endl;
    ini << "slr=relay_SK2_rout:SLR1" << std::endl;

    // SpMV sp tags
    // TODO: parameterize cluster allocation
    const unsigned SLR_0_CLUSTERS = 4;
    const unsigned SLR_1_CLUSTERS = 6;
    const unsigned SLR_2_CLUSTERS = 6;
    for (size_t hbm_idx = 0; hbm_idx < this->num_channels_; hbm_idx++) {
        ini << "sp=spmv_sk";
        if (hbm_idx < SLR_0_CLUSTERS) {
            ini << 0 << "_1";
        } else if (hbm_idx < SLR_0_CLUSTERS + SLR_1_CLUSTERS) {
            ini << 1 << "_1";
        } else {
            ini << 2 << "_1";
        }
        ini << ".matrix_hbm_" << hbm_idx << ":HBM["
            << hbm_idx << "]" << std::endl;
    }
    ini << "sp=VL.packed_dense_vector:HBM[20]" << std::endl;
    ini << "sp=RD.packed_dense_mask:HBM[21]" << std::endl;
    ini << "sp=RD.packed_dense_result:HBM[22]" << std::endl;

    // SpMV sc tags
    // TODO: parameterize AXIS FIFO depth
    ini << "sc=VL.to_SLR0:spmv_sk0_1.vec_in:32" << std::endl;
    ini << "sc=VL.to_SLR1:spmv_sk1_1.vec_in:32" << std::endl;
    ini << "sc=VL.to_SLR2:relay_SK2_vin.in:32" << std::endl;
    ini << "sc=relay_SK2_vin.out:spmv_sk2_1.vec_in:32" << std::endl;
    ini << "sc=spmv_sk0_1.res_out:RD.from_SLR0:32" << std::endl;
    ini << "sc=spmv_sk1_1.res_out:RD.from_SLR1:32" << std::endl;
    ini << "sc=spmv_sk2_1.res_out:relay_SK2_rout.in:32" << std::endl;
    ini << "sc=relay_SK2_rout.out:RD.from_SLR2:32" << std::endl;

    // SpMSpV apply overlay slr tags
    ini << "slr=spmspv_apply_1:SLR0" << std::endl;

    // SpMSpV apply overlay sp tags
    ini << "sp=spmspv_apply_1.spmv_vector:HBM[20]" << std::endl;
    ini << "sp=spmspv_apply_1.spmv_mask:HBM[21]" << std::endl;
    ini << "sp=spmspv_apply_1.spmv_mask_w:HBM[21]" << std::endl;
    ini << "sp=spmspv_apply_1.spmv_out:HBM[22]" << std::endl;

    ini << "sp=spmspv_apply_1.spmspv_matrix_0:HBM[23]" << std::endl;
    ini << "sp=spmspv_apply_1.spmspv_matrix_indptr_0:HBM[23]" << std::endl;
    ini << "sp=spmspv_apply_1.spmspv_matrix_partptr_0:HBM[23]" << std::endl;

    ini << "sp=spmspv_apply_1.spmspv_matrix_1:HBM[24]" << std::endl;
    ini << "sp=spmspv_apply_1.spmspv_matrix_indptr_1:HBM[24]" << std::endl;
    ini << "sp=spmspv_apply_1.spmspv_matrix_partptr_1:HBM[24]" << std::endl;

    ini << "sp=spmspv_apply_1.spmspv_vector:HBM[20]" << std::endl;
    ini << "sp=spmspv_apply_1.spmspv_mask:HBM[21]" << std::endl;
    ini << "sp=spmspv_apply_1.spmspv_out:HBM[22]" << std::endl;

    // enable retiming
    /* retiming will be automatically enabled
       if we build with "--optimize 3"
    */
    // ini << "[vivado]" << std::endl;
    // ini << "prop=run.__KERNEL__.{STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS}={-retiming}" << std::endl;
    ini.close();
}


}  // namespace synthesizer
}  // namespace graphlily

#endif  // GRAPHLILY_OVERLAY_SYNTHESIZER_H_
