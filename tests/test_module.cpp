#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include "ap_fixed.h"
#include "graphblas/base.h"
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
    std::string common_path = "/work/shared/common/research/graphblas/";
    std::string csr_float_npz_path = common_path + "data/sparse_matrix_graph/uniform_10K_10_csr_float32.npz";
    std::string xclbin_file_path = common_path + "bitstreams/spmv/spmv_bool_ufixed.32.1_MulAdd.xclbin";
    std::string kernel_name = "kernel_spmv_v4";

    graphblas::SemiRingType semiring = graphblas::kMulAdd;
    uint32_t num_channels = 2;
    uint32_t vector_buffer_len = 10000;
    using matrix_data_t = bool;
    using vector_data_t = ap_ufixed<32,1>;
    graphblas::module::SpMVModule<matrix_data_t, vector_data_t> module(csr_float_npz_path, semiring,
                                                                       num_channels, vector_buffer_len,
                                                                       xclbin_file_path, kernel_name);
    uint32_t num_cols = 10000;
    std::vector<float, aligned_allocator<float>> vector_float(num_cols);
    std::generate(vector_float.begin(), vector_float.end(), [&](){return float(rand() % num_cols) / num_cols / num_cols;});
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> vector(vector_float.begin(), vector_float.end());

    std::vector<vector_data_t, aligned_allocator<vector_data_t>> kernel_results =
        module.run(vector);
    std::vector<float, aligned_allocator<float>> reference_results =
        module.compute_reference_results(vector_float);

    verify<vector_data_t>(reference_results, kernel_results);
    std::cout << "Test passed" << std::endl;
}


int main(int argc, char *argv[]) {
    test_spmv_module();
}

#pragma GCC diagnostic pop
