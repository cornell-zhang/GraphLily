#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include "graphlily/io/data_loader.h"
#include "graphlily/io/data_formatter.h"

#include <chrono>
#include <iostream>

using namespace graphlily::io;


size_t count_bytes_cpsr_matrix(CPSRMatrix<float, graphlily::pack_size>& cpsr_matrix) {
    size_t n_bytes = 0;
    for (auto x: cpsr_matrix.formatted_adj_data) {
        n_bytes += x.size() * 4 * graphlily::pack_size;
    }
    for (auto x: cpsr_matrix.formatted_adj_indices) {
        n_bytes += x.size() * 4 * graphlily::pack_size;
    }
    for (auto x: cpsr_matrix.formatted_adj_indptr) {
        n_bytes += x.size() * 4 * graphlily::pack_size;
    }
    return n_bytes;
}


void bench_preprocess(uint32_t num_channels, uint32_t out_buf_len, uint32_t vec_buf_len, std::string dataset) {
    std::string csr_float_npz_path = dataset;
    CSRMatrix<float> csr_matrix = load_csr_matrix_from_float_npz(csr_float_npz_path);
    util_round_csr_matrix_dim(csr_matrix, num_channels*graphlily::pack_size, graphlily::pack_size);

    /********************************************************/
    auto start = std::chrono::high_resolution_clock::now();

    bool skip_empty_rows = true;
    CPSRMatrix<float, graphlily::pack_size> cpsr_matrix = csr2cpsr<float, graphlily::pack_size>(
        csr_matrix,
        graphlily::idx_marker,
        out_buf_len,
        vec_buf_len,
        num_channels,
        skip_empty_rows);

    auto end = std::chrono::high_resolution_clock::now();
    float time_in_ms = float(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000;
    std::cout << "skip_empty_rows is true, time: " << time_in_ms << " ms" << std::endl;

    /********************************************************/
    start = std::chrono::high_resolution_clock::now();

    skip_empty_rows = false;
    cpsr_matrix = csr2cpsr<float, graphlily::pack_size>(
        csr_matrix,
        graphlily::idx_marker,
        out_buf_len,
        vec_buf_len,
        num_channels,
        skip_empty_rows);

    end = std::chrono::high_resolution_clock::now();
    time_in_ms = float(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000;
    std::cout << "skip_empty_rows is false, time: " << time_in_ms << " ms" << std::endl;

    /********************************************************/
    skip_empty_rows = true;
    cpsr_matrix = csr2cpsr<float, graphlily::pack_size>(
        csr_matrix,
        graphlily::idx_marker,
        out_buf_len,
        vec_buf_len,
        num_channels,
        skip_empty_rows);
    size_t total_bytes = count_bytes_cpsr_matrix(cpsr_matrix);
    size_t nnz = csr_matrix.adj_indptr[csr_matrix.num_rows];
    float ratio = total_bytes / float(8 * nnz);
    std::cout << "skip_empty_rows is true" << std::endl;
    std::cout << "total_bytes: " << total_bytes << std::endl;
    std::cout << "nnz: " << nnz << std::endl;
    std::cout << "ratio: " << ratio << std::endl;

    skip_empty_rows = false;
    cpsr_matrix = csr2cpsr<float, graphlily::pack_size>(
        csr_matrix,
        graphlily::idx_marker,
        out_buf_len,
        vec_buf_len,
        num_channels,
        skip_empty_rows);
    total_bytes = count_bytes_cpsr_matrix(cpsr_matrix);
    nnz = csr_matrix.adj_indptr[csr_matrix.num_rows];
    ratio = total_bytes / float(8 * nnz);
    std::cout << "skip_empty_rows is false" << std::endl;
    std::cout << "total_bytes: " << total_bytes << std::endl;
    std::cout << "nnz: " << nnz << std::endl;
    std::cout << "ratio: " << ratio << std::endl;
}


int main(int argc, char *argv[]) {
    bench_preprocess(strtol(argv[1], NULL, 10),
                     strtol(argv[2], NULL, 10),
                     strtol(argv[3], NULL, 10),
                     argv[4]);
}

#pragma GCC diagnostic pop
