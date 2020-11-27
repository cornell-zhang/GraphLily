#include <iostream>
#include <chrono>

#include "xcl2.hpp"

#include "graphlily/io/data_loader.h"
#include "graphlily/module/spmv_module.h"


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


void bench_spmv(uint32_t num_channels, std::string bitstream, std::string dataset) {
    uint32_t out_buf_len = 1000 * 1024;
    uint32_t vec_buf_len = 30 * 1024;
    graphlily::module::SpMVModule<graphlily::val_t, graphlily::val_t> spmv(num_channels,
                                                                           out_buf_len,
                                                                           vec_buf_len);
    spmv.set_target("hw");
    spmv.set_up_runtime(bitstream);
    spmv.set_semiring(graphlily::ArithmeticSemiring);
    spmv.set_mask_type(graphlily::kNoMask);

    std::string csr_float_npz_path = dataset;
    CSRMatrix<float> csr_matrix = graphlily::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
    for (auto &x : csr_matrix.adj_data) x = 1;

    graphlily::io::util_round_csr_matrix_dim(
        csr_matrix,
        num_channels * graphlily::pack_size * graphlily::num_cycles_float_add,
        graphlily::pack_size * graphlily::num_cycles_float_add);

    std::vector<float, aligned_allocator<float>> vector_float(csr_matrix.num_cols);
    std::generate(vector_float.begin(), vector_float.end(), [&]{return float(rand() % 2);});
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> vector(vector_float.begin(),
                                                                              vector_float.end());

    std::vector<float, aligned_allocator<float>> mask_float(csr_matrix.num_cols);
    std::generate(mask_float.begin(), mask_float.end(), [&](){return float(rand() % 2);});
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> mask(mask_float.begin(),
                                                                            mask_float.end());

    bool skip_empty_rows = true;
    spmv.load_and_format_matrix(csr_matrix, skip_empty_rows);

    std::cout << "finished load_and_format_matrix" << std::endl;

    spmv.send_matrix_host_to_device();
    spmv.send_vector_host_to_device(vector);
    spmv.send_mask_host_to_device(mask);

    std::cout << "start run" << std::endl;

    spmv.run();

    uint32_t num_runs = 100;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        spmv.run();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    float average_time_in_sec = float(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())
        / 1000000 / num_runs;
    std::cout << "average_time: " << average_time_in_sec * 1000 << " ms" << std::endl;

    uint32_t nnz = spmv.get_nnz();
    double throughput = nnz * (sizeof(unsigned) + sizeof(unsigned)); // indices + values
    throughput /= 1000;                // to KB
    throughput /= 1000;                // to MB
    throughput /= 1000;                // to GB
    throughput /= average_time_in_sec; // to GB/s
    // std::cout << "Memory THROUGHPUT = " << throughput << " GB/s" << std::endl;
    std::cout << "Compute THROUGHPUT = " << throughput / (sizeof(unsigned) + sizeof(unsigned))
              << " GOPS" << std::endl;

    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> kernel_results =
        spmv.send_results_device_to_host();
    std::vector<float, aligned_allocator<float>> reference_results =
        spmv.compute_reference_results(vector_float);
    verify<graphlily::val_t>(reference_results, kernel_results);
    std::cout << "SpMV passed" << std::endl;
}


int main(int argc, char *argv[]) {
    bench_spmv(strtol(argv[1], NULL, 10), argv[2], argv[3]);
}
