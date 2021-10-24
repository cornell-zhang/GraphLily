#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <iostream>
#include <chrono>

#include "graphlily/app/pagerank.h"


template<typename data_t>
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


void bench_pagerank(uint32_t num_channels, uint32_t spmv_out_buf_len,
                    uint32_t vec_buf_len, std::string bitstream, std::string dataset) {
    graphlily::app::PageRank pagerank(graphlily::num_hbm_channels, spmv_out_buf_len, vec_buf_len);
    pagerank.set_target("hw");
    pagerank.set_up_runtime(bitstream);

    float damping = 0.9;
    bool skip_empty_rows = true;
    pagerank.load_and_format_matrix(dataset, damping, skip_empty_rows);
    std::cout << "finished load_and_format_matrix" << std::endl;
    pagerank.send_matrix_host_to_device();

    uint32_t num_iterations = 10;
    auto reference_results = pagerank.compute_reference_results(damping, num_iterations);

    auto kernel_results = pagerank.pull(damping, num_iterations);
    // verify<graphlily::val_t>(reference_results, kernel_results);
    // std::cout << "PageRank passed" << std::endl;

    uint32_t num_runs = 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        kernel_results = pagerank.pull(damping, num_iterations);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    float average_time_in_sec = float(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())
        / 1000000 / num_runs / num_iterations;
    std::cout << "PageRank time for one iteration: " << average_time_in_sec * 1000 << " ms" << std::endl;
    uint32_t nnz = pagerank.get_nnz();
    double op_count = nnz;
    double throughput = op_count / 1000 / 1000 / 1000 / average_time_in_sec;
    std::cout << "PageRank Compute THROUGHPUT = " << throughput << " GTEPS" << std::endl;
}


int main(int argc, char *argv[]) {
    bench_pagerank(strtol(argv[1], NULL, 10),
                   strtol(argv[2], NULL, 10),
                   strtol(argv[3], NULL, 10),
                   argv[4],
                   argv[5]);
}

#pragma GCC diagnostic pop
