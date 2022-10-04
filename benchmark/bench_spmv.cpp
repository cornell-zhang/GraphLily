#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <iostream>
#include <chrono>

#include "xcl2.hpp"

#include "graphlily/io/data_loader.h"
#include "graphlily/module/spmv_module.h"


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


void bench_spmv(uint32_t num_channels, uint32_t out_buf_len, uint32_t vec_buf_len,
                std::string bitstream, std::string dataset) {
    graphlily::module::SpMVModule<graphlily::val_t, graphlily::val_t> spmv(num_channels,
                                                                           out_buf_len,
                                                                           vec_buf_len);
    spmv.set_target("hw");
    graphlily::MaskType mask_type = graphlily::kNoMask;
    spmv.set_mask_type(mask_type);
    spmv.set_semiring(graphlily::ArithmeticSemiring);
    spmv.set_up_runtime(bitstream);

    std::string csr_float_npz_path = dataset;
    CSRMatrix<float> csr_matrix = graphlily::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
    for (auto &x : csr_matrix.adj_data) x = 1.0 / csr_matrix.num_rows;

    graphlily::io::util_round_csr_matrix_dim(
        csr_matrix,
        num_channels * graphlily::pack_size,
        graphlily::pack_size);

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
    // send the mask to device even if the kernel does not use it
    spmv.send_mask_host_to_device(mask);

    std::cout << "start run" << std::endl;

    spmv.run();

    auto kernel_results = spmv.send_results_device_to_host();
    std::vector<float, aligned_allocator<float>> reference_results;
    if (mask_type == graphlily::kNoMask) {
        reference_results = spmv.compute_reference_results(vector_float);
    } else {
        reference_results = spmv.compute_reference_results(vector_float, mask_float);
    }

    // for (int i = 0; i < 10; i++) {
    //     std::cout << reference_results[i] << " " << kernel_results[i] <<std::endl;
    // }

    // verify<graphlily::val_t>(reference_results, kernel_results);
    // std::cout << "SpMV passed" << std::endl;

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
    double throughput = nnz;
    throughput /= 1000;
    throughput /= 1000;
    throughput /= 1000;
    throughput /= average_time_in_sec;
    std::cout << "Compute THROUGHPUT = " << throughput << " GTEPS" << std::endl;
}


int main(int argc, char *argv[]) {
    bench_spmv(strtol(argv[1], NULL, 10),
               strtol(argv[2], NULL, 10),
               strtol(argv[3], NULL, 10),
               argv[4],
               argv[5]);
}

#pragma GCC diagnostic pop
