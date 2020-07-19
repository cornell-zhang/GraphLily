#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include "ap_fixed.h"
#include "graphblas/global.h"
#include "graphblas/module/spmv_module.h"


template <typename data_t>
void verify(std::vector<float, aligned_allocator<float>> &reference_results,
            std::vector<data_t, aligned_allocator<data_t>> &kernel_results) {
    if (!(reference_results.size() == kernel_results.size())) {
        std::cout << "Size mismatch!" << std::endl;
        exit(EXIT_FAILURE);
    }
    float epsilon = 0.0001;
    for (size_t i = 0; i < reference_results.size(); i++) {
        if (abs(float(kernel_results[i]) - reference_results[i]) > epsilon) {
            std::cout << "Error: Result mismatch"
                      << std::endl;
            std::cout << "i = " << i
                      << " Reference result = " << reference_results[i]
                      << " Kernel result = " << kernel_results[i]
                      << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}


void test_spmv_module() {
    std::string csr_float_npz_path = "/work/shared/common/research/graphblas/"
                                     "data/sparse_matrix_graph/uniform_10K_10_csr_float32.npz";
    graphblas::SemiRingType semiring = graphblas::kLogicalAndOr;
    uint32_t num_channels = 2;
    uint32_t vector_buffer_len = 10000;
    std::string kernel_name = "kernel_spmv";
    using matrix_data_t = bool;
    using vector_data_t = unsigned int; // Use unsigned int to work around the issue with std::vector<bool>
    graphblas::module::SpMVModule<matrix_data_t, vector_data_t> module(csr_float_npz_path, semiring,
                                                                       num_channels, vector_buffer_len,
                                                                       kernel_name);
    std::string target = "sw_emu";
    module.set_target(target);
    module.compile();
    module.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");
    module.send_data_to_FPGA();
    uint32_t num_cols = 10000;
    std::vector<float, aligned_allocator<float>> vector_float(num_cols);
    std::generate(vector_float.begin(), vector_float.end(), [&](){return float(rand() % 2);});
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> vector(vector_float.begin(), vector_float.end());

    std::vector<vector_data_t, aligned_allocator<vector_data_t>> kernel_results = module.run(vector);
    std::vector<float, aligned_allocator<float>> reference_results = module.compute_reference_results(vector_float);
    verify<vector_data_t>(reference_results, kernel_results);
    std::cout << "Test passed" << std::endl;
}


int main(int argc, char *argv[]) {
    test_spmv_module();
}

#pragma GCC diagnostic pop
