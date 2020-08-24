#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <ap_fixed.h>
#include <iostream>

#include "graphblas/app/bfs.h"
#include "graphblas/app/pagerank.h"


std::string target = "sw_emu";


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
    uint32_t num_channels = 8;
    uint32_t out_buffer_len = 1024;
    uint32_t vector_buffer_len = 1024;
    graphblas::app::BFS bfs(num_channels, out_buffer_len, vector_buffer_len);

    bfs.set_target(target);
    bfs.compile();
    bfs.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    std::string csr_float_npz_path = "/work/shared/common/research/graphblas/"
                                     "data/sparse_matrix_graph/uniform_10K_10_csr_float32.npz";
    bfs.load_and_format_matrix(csr_float_npz_path);
    bfs.send_matrix_host_to_device();

    uint32_t source = 0;
    uint32_t num_iterations = 10;
    auto kernel_results = bfs.run(source, num_iterations);
    auto reference_results = bfs.compute_reference_results(source, num_iterations);
    verify<unsigned int>(reference_results, kernel_results);
    std::cout << "BFS test passed" << std::endl;
}


void test_pagerank() {
    uint32_t num_channels = 8;
    uint32_t out_buffer_len = 1024;
    uint32_t vector_buffer_len = 1024;
    graphblas::app::PageRank pagerank(num_channels, out_buffer_len, vector_buffer_len);

    float damping = 0.9;
    uint32_t num_iterations = 10;

    pagerank.set_target(target);
    pagerank.compile();
    pagerank.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    std::string csr_float_npz_path = "/work/shared/common/research/graphblas/"
                                     "data/sparse_matrix_graph/uniform_10K_10_csr_float32.npz";
    pagerank.load_and_format_matrix(csr_float_npz_path, damping);
    pagerank.send_matrix_host_to_device();

    auto kernel_results = pagerank.run(damping, num_iterations);
    auto reference_results = pagerank.compute_reference_results(damping, num_iterations);
    verify<ap_ufixed<32, 1>>(reference_results, kernel_results);
    std::cout << "PageRank test passed" << std::endl;
}


int main(int argc, char *argv[]) {
    // Cannot run more than one application, causing runtime error: some device is already programmed
    // clean_proj_folder();
    // test_bfs();

    clean_proj_folder();
    test_pagerank();
}

#pragma GCC diagnostic pop
