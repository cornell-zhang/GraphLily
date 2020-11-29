#include <iostream>
#include <chrono>

#include "graphlily/app/bfs.h"


void bench_bfs(uint32_t num_channels, std::string bitstream, std::string dataset) {
    uint32_t out_buf_len = 320 * 1024;
    uint32_t vec_buf_len = 64 * 1024;
    graphlily::app::BFS bfs(num_channels, out_buf_len, vec_buf_len);

    std::string target = "hw";
    bfs.set_target(target);
    bfs.set_up_runtime(bitstream);

    bfs.load_and_format_matrix(dataset);
    bfs.send_matrix_host_to_device();

    uint32_t source = 0;
    uint32_t num_iterations = 10;
    auto kernel_results = bfs.push(source, num_iterations);
    uint32_t num_runs = 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        kernel_results = bfs.push(source, num_iterations);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    float average_time_in_sec = float(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())
        / 1000000 / num_runs;
    std::cout << "average_time_in_sec = " << average_time_in_sec << std::endl;

    uint32_t nnz = bfs.get_nnz();
    double throughput = num_iterations * nnz * sizeof(unsigned); // indices
    throughput /= 1000;                // to KB
    throughput /= 1000;                // to MB
    throughput /= 1000;                // to GB
    throughput /= average_time_in_sec; // to GB/s
    // std::cout << "Memory THROUGHPUT = " << throughput << " GB/s" << std::endl;
    std::cout << "Compute THROUGHPUT = " << throughput / sizeof(unsigned) << " GOPS" << std::endl;
}


int main(int argc, char *argv[]) {
    bench_bfs(strtol(argv[1], NULL, 10), argv[2], argv[3]);
}
