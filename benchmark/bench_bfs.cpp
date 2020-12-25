#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <iostream>
#include <chrono>

#include "graphlily/app/bfs.h"


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


void bench_bfs(uint32_t num_channels, uint32_t spmv_out_buf_len,
               uint32_t spmspv_out_buf_len, uint32_t vec_buf_len,
               std::string bitstream, std::string dataset) {
    graphlily::app::BFS bfs(num_channels, spmv_out_buf_len, spmspv_out_buf_len, vec_buf_len);
    bfs.set_target("hw");
    bfs.set_up_runtime(bitstream);

    bool skip_empty_rows = true;
    bfs.load_and_format_matrix(dataset, skip_empty_rows);
    std::cout << "finished load_and_format_matrix" << std::endl;
    bfs.send_matrix_host_to_device();

    uint32_t source = 0;
    uint32_t num_iterations = 10;
    auto reference_results = bfs.compute_reference_results(source, num_iterations);

    // Make sure the results make sense, e.g., the starting vertex connects to at least one vertex
    for (int i = 0; i < 10; i++) {
        std::cout << reference_results[i] <<std::endl;
    }

    // Pull
    auto kernel_results = bfs.pull(source, num_iterations);
    verify<graphlily::val_t>(reference_results, kernel_results);
    std::cout << "BFS pull passed" << std::endl;

    uint32_t num_runs = 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        kernel_results = bfs.pull(source, num_iterations);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    float average_time_in_sec = float(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())
        / 1000000 / num_runs;
    std::cout << "Pull average_time_in_sec = " << average_time_in_sec << std::endl;
    uint32_t nnz = bfs.get_nnz();
    double op_count = nnz * num_iterations;
    double throughput = op_count / 1000 / 1000 / 1000 / average_time_in_sec;
    std::cout << "Pull Compute THROUGHPUT = " << throughput << " GOPS" << std::endl;

    // Pull-Push
    float threshold = 0.01;
    kernel_results = bfs.pull_push(source, num_iterations, threshold);
    verify<graphlily::val_t>(reference_results, kernel_results);
    std::cout << "BFS pull-push passed" << std::endl;

    num_runs = 1;
    t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        kernel_results = bfs.pull_push(source, num_iterations, threshold);
    }
    t2 = std::chrono::high_resolution_clock::now();
    average_time_in_sec = float(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())
        / 1000000 / num_runs;
    std::cout << "Pull-Push average_time_in_sec = " << average_time_in_sec << std::endl;
    throughput = op_count / 1000 / 1000 / 1000 / average_time_in_sec;
    std::cout << "Pull-Push Compute THROUGHPUT = " << throughput << " GOPS" << std::endl;
}


int main(int argc, char *argv[]) {
    bench_bfs(strtol(argv[1], NULL, 10),
              strtol(argv[2], NULL, 10),
              strtol(argv[3], NULL, 10),
              strtol(argv[4], NULL, 10),
              argv[5],
              argv[6]);
}

#pragma GCC diagnostic pop
