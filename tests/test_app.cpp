#include <iostream>

#include "graphblas/app/bfs.h"


void clean_proj_folder() {
    std::string command = "rm -rf ./" + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
}


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


void test_bfs() {
    std::string csr_float_npz_path = "/work/shared/common/research/graphblas/"
                                     "data/sparse_matrix_graph/uniform_10K_10_csr_float32.npz";
    graphblas::app::BFS bfs(csr_float_npz_path);
    std::string target = "sw_emu";
    bfs.set_target(target);
    bfs.compile();
    bfs.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");
    bfs.send_data_to_FPGA();

    uint32_t source = 0;
    uint32_t num_iterations = 10;
    auto kernel_results = bfs.run(source, num_iterations);
    auto reference_results = bfs.compute_reference_results(source, num_iterations);
    verify<unsigned int>(reference_results, kernel_results);
    std::cout << "BFS test passed" << std::endl;
}


int main(int argc, char *argv[]) {
    clean_proj_folder();
    test_bfs();
}
