#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>

#include "graphlily/io/data_loader.h"
#include "graphlily/module/spmspv_module.h"

std::string line(int len) {
    std::string l;
    for (size_t i = 0; i < len; i++) {
        l += "-";
    }
    return l;
}

// Matrix dataset folder
const std::string matrix_dataset_dir = "/work/shared/common/research/graphblas/data/sparse_matrix_graph/";

// verify results
template <typename DataT>
bool verify(std::vector<float, aligned_allocator<float>> &reference_results,
            std::vector<DataT, aligned_allocator<DataT>> &kernel_results) {
    if (!(reference_results.size() == kernel_results.size())) {
        std::cout << "Size mismatch!" << std::endl;
        return false;
    }
    float epsilon = 1;
    for (size_t i = 0; i < reference_results.size(); i++) {
        if (abs(float(kernel_results[i]) - reference_results[i]) > epsilon) {
            std::cout << "Error: Result mismatch"
                      << std::endl;
            std::cout << "i = " << i
                      << " Reference result = " << reference_results[i]
                      << " Kernel result = " << kernel_results[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

// measure data usage (in Bytes, only measure the matrix)
double measure_data_usage(graphlily::io::CSCMatrix<float> &matrix,
                          graphlily::aligned_sparse_float_vec_t &vector) {
    double data_usage = 0;
    unsigned vec_nnz_total = vector[0].index;

    // loop over all active columns
    for (unsigned active_colid = 0; active_colid < vec_nnz_total; active_colid++) {
        graphlily::idx_t current_colid = vector[active_colid + 1].index;
        // slice out the current column out of the active columns
        graphlily::idx_t col_start = matrix.adj_indptr[current_colid];
        graphlily::idx_t col_end = matrix.adj_indptr[current_colid + 1];
        data_usage += (sizeof(graphlily::val_t) + sizeof(graphlily::idx_t)) * (col_end - col_start);
    }

    return data_usage;
}

struct BenchCaseInfo {
    std::string name;
    std::string semiring_str;
    std::string spv_str;
};

struct BenchCase {
    BenchCaseInfo info;
    graphlily::SemiringType semiring;
    graphlily::MaskType mask_type;
    std::string matrix_path;
    double vector_sparsity;
};

BenchCase set_up_bench_case(std::string matrix_file_name,
                            graphlily::SemiringType semiring,
                            graphlily::MaskType mask_type,
                            double vector_sparsity) {
    BenchCase bcase;
    if (matrix_file_name.length() > 35) {
        bcase.info.name = matrix_file_name.substr(0, 33) + "...";
    } else {
        bcase.info.name = matrix_file_name;
    }

    bcase.semiring = semiring;
    switch (semiring.op) {
    case graphlily::kMulAdd:
        bcase.info.semiring_str = "Arithmetic";
        break;
    case graphlily::kLogicalAndOr:
        bcase.info.semiring_str = "Logical";
        break;
    case graphlily::kAddMin:
        bcase.info.semiring_str = "Tropical";
        break;
    default:
        bcase.info.semiring_str = "UNSUPPORTED";
        break;
    }

    bcase.matrix_path = matrix_dataset_dir + matrix_file_name;

    bcase.mask_type = mask_type;

    bcase.vector_sparsity = vector_sparsity;
    char c[11];
    sprintf(c, "%3.5f%%", vector_sparsity * 100);
    bcase.info.spv_str.assign(c);

    return bcase;
}

// time_elapsed < 0 when test not passed.
struct BenchCaseResult {
    BenchCaseInfo info;
    double avg_time_ms;
    double avg_GOPS;
    double avg_thpt_GBPS;
};

BenchCaseResult run_bench_case(graphlily::module::SpMSpVModule<graphlily::val_t,
                               graphlily::val_t, graphlily::idx_val_t> &spmspv,
                               BenchCase benchmark_case) {
    BenchCaseResult benchmark_result;
    benchmark_result.info = benchmark_case.info;
    std::cout << "INFO: [Benchmark] "
              << benchmark_result.info.name << " "
              << benchmark_result.info.spv_str << " "
              << benchmark_result.info.semiring_str << std::endl;

    using aligned_sparse_vector_t = std::vector<graphlily::idx_val_t, aligned_allocator<graphlily::idx_val_t>>;
    using aligned_dense_vector_t = std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>>;

    graphlily::io::CSCMatrix<float> matrix = graphlily::io::csr2csc<float>(
        graphlily::io::load_csr_matrix_from_float_npz(benchmark_case.matrix_path));
    for (auto &x : matrix.adj_data) x = 0.5;

    // generate vector
    unsigned int num_vec_nnz = floor((1 - benchmark_case.vector_sparsity) * matrix.num_cols);
    unsigned int ind_incr = matrix.num_cols / num_vec_nnz;
    aligned_sparse_vector_t vector(num_vec_nnz + 1);
    graphlily::aligned_sparse_float_vec_t vector_float(num_vec_nnz + 1);
    vector[0].index = num_vec_nnz;
    vector_float[0].index = num_vec_nnz;
    for (int i = 1; i < num_vec_nnz + 1; i++) {
        float v = (rand() % 99 + 1) / 100.0;
        switch (benchmark_case.semiring.op) {
            case graphlily::kMulAdd:
                vector[i].val = (graphlily::val_t)v;
                vector_float[i].val = v;
            break;
            case graphlily::kLogicalAndOr:
                vector[i].val = (graphlily::val_t)1;
                vector_float[i].val = 1;
            break;
            case graphlily::kAddMin:
                vector[i].val = (graphlily::val_t)v;
                vector_float[i].val = v;
            break;
            default:
                vector[i].val = benchmark_case.semiring.zero;
                vector_float[i].val = benchmark_case.semiring.zero;
            break;
        }
        vector[i].index = (i - 1) * ind_incr;
        vector_float[i].index = (i - 1) * ind_incr;
    }

    // generate mask
    aligned_dense_vector_t mask(matrix.num_cols);
    graphlily::aligned_dense_float_vec_t mask_float(matrix.num_cols);
    for (int i = 0; i < matrix.num_cols; i++) {
        float m = (rand() % 2) ? (graphlily::val_t)1 : benchmark_case.semiring.zero;
        mask[i] = (graphlily::val_t)m;
        mask_float[i] = m;
    }

    spmspv.config_kernel(benchmark_case.semiring, benchmark_case.mask_type);
    spmspv.load_and_format_matrix(matrix);
    spmspv.send_matrix_host_to_device();
    spmspv.send_vector_host_to_device(vector);
    spmspv.send_mask_host_to_device(mask);

    std::cout << "INFO: [Benchmark] Warm-up run and verify" << std::endl;
    spmspv.run();
    aligned_sparse_vector_t kernel_results = spmspv.send_results_device_to_host();

    // convert sparse vector to dense vector
    aligned_dense_vector_t kernel_results_dense(matrix.num_rows, benchmark_case.semiring.zero);
    for (size_t i = 1; i < kernel_results[0].index + 1; i++) {
        kernel_results_dense[kernel_results[i].index] = kernel_results[i].val;
    }

    graphlily::aligned_dense_float_vec_t reference_result =
        spmspv.compute_reference_results(vector_float, mask_float);

    bool passed = verify<graphlily::val_t>(reference_result, kernel_results_dense);
    // bool passed = true;
    if (!passed) {
        std::cout << "ERROR: [Benchmark] Result mismatch! Aborting..." << std::endl;
        benchmark_result.avg_time_ms = -1;
        benchmark_result.avg_GOPS = -1;
        benchmark_result.avg_thpt_GBPS = -1;
        return benchmark_result;
    }
    std::cout << "INFO: [Benchmark] Warm-up passed. Benchmark Start" << std::endl;

    double data_usage = measure_data_usage(matrix, vector_float);

    int num_runs = 20;
    double time_in_us = 0;
    for (size_t i = 0; i < num_runs; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        spmspv.run();
        auto t2 = std::chrono::high_resolution_clock::now();
        time_in_us += double(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    }
    benchmark_result.avg_time_ms = time_in_us / num_runs / 1000;
    benchmark_result.avg_thpt_GBPS = data_usage / 1000 / 1000 / benchmark_result.avg_time_ms;
    benchmark_result.avg_GOPS = benchmark_result.avg_thpt_GBPS / (sizeof(graphlily::val_t)
        + sizeof(graphlily::idx_t));

    return benchmark_result;
}


int main(int argc, char *argv[]) {
    if (!(argc == 4 || argc == 3)) {
        std::cout << "Usage:" << std::endl;
        std::cout << "  to run benchmark: "<< argv[0] << " <target> <xclbin> <logfile>" << std::endl;
        std::cout << "  to build kernel: "<< argv[0] << " <target> build" << std::endl;
        std::cout << "Aborting..." << std::endl;
        return 0;
    }
    std::string target = argv[1];
    std::string xclbin = argv[2];

    std::vector<std::string> matrix_set;
    std::vector<float> vector_sparsity_set;
    std::vector<graphlily::SemiringType> semiring_set;

    if (target == "hw_emu") {
        matrix_set.push_back("gplus_108K_13M_csr_float32.npz");
        vector_sparsity_set.push_back(0.99);
    } else {
        matrix_set.push_back("gplus_108K_13M_csr_float32.npz");
        matrix_set.push_back("reddit_233K_115M_csr_float32.npz");
        matrix_set.push_back("ogbn_proteins_132K_79M_csr_float32.npz");
        matrix_set.push_back("ogbl_ppa_576K_42M_csr_float32.npz");
        matrix_set.push_back("rMat_64K_64_csr_float32.npz");
        matrix_set.push_back("rMat_64K_256_csr_float32.npz");
        matrix_set.push_back("rMat_256K_64_csr_float32.npz");
        matrix_set.push_back("rMat_256K_256_csr_float32.npz");

        vector_sparsity_set.push_back(0.90);
        vector_sparsity_set.push_back(0.95);
        vector_sparsity_set.push_back(0.99);
        vector_sparsity_set.push_back(0.995);
        vector_sparsity_set.push_back(0.999);
        vector_sparsity_set.push_back(0.9995);
        vector_sparsity_set.push_back(0.9999);
    }

    semiring_set.push_back(graphlily::ArithmeticSemiring);
    // semiring_set.push_back(graphlily::LogicalSemiring);

    std::vector<BenchCaseResult> benchmark_results;

    uint32_t out_buf_len = 256 * 1024;
    graphlily::module::SpMSpVModule<graphlily::val_t, graphlily::val_t, graphlily::idx_val_t>
        spmspv_module(out_buf_len);

    spmspv_module.set_target(target);
    if (xclbin == "build") {
        std::string command = "rm -rf ./" + graphlily::proj_folder_name;
        std::cout << command << std::endl;
        system(command.c_str());
        spmspv_module.compile();
        std::cout << "Kernel Build Complete" << std::endl;
        return 0;
    }

    std::string logfile = argv[3];
    std::ofstream result_log_file(logfile);

    spmspv_module.set_up_runtime(xclbin);

    for (size_t mat_id = 0; mat_id < matrix_set.size(); mat_id++) {
        for (size_t semiring_id = 0; semiring_id < semiring_set.size(); semiring_id++) {
            for (size_t vec_sp_id = 0; vec_sp_id < vector_sparsity_set.size(); vec_sp_id++) {
                benchmark_results.push_back(
                    run_bench_case(
                        spmspv_module,
                        set_up_bench_case(
                            matrix_set[mat_id],
                            semiring_set[semiring_id],
                            graphlily::kNoMask,
                            vector_sparsity_set[vec_sp_id]
                        )
                    )
                );
            }
        }
    }

    bool all_passed = true;
    for (size_t i = 0; i < benchmark_results.size(); i++) {
        all_passed = all_passed && (benchmark_results[i].avg_time_ms >= 0);
    }

    // print results
    std::cout << std::endl;
    std::cout << "INFO: [Benckmark] Verification result: " << (all_passed?"PASSED":"FAILED") << std::endl;
    std::cout << "  Details:" << std::endl;
    std::cout << "  "
                << std::setw(50) << "test case"
                << std::setw(20) << "semiring"
                << std::setw(20) << "vector sparsity"
                << std::setw(10) << "result"
                << std::setw(20) << "time(ms)"
                << std::setw(20) << "performance(GOPS)"
                << std::setw(20) << "throughput(GB/s)"
                << std::endl;
    std::cout << "  " << line(50 + 20 + 20 + 10 + 20 + 20 + 20) << std::endl;

    for (size_t i = 0; i < benchmark_results.size(); i++) {
        std::cout << "  "
                << std::setw(50) << benchmark_results[i].info.name
                << std::setw(20) << benchmark_results[i].info.semiring_str
                << std::setw(20) << benchmark_results[i].info.spv_str
                << std::setw(10) << ((benchmark_results[i].avg_time_ms >= 0) ? "PASSED" : "FAILED")
                << std::setw(20) << benchmark_results[i].avg_time_ms
                << std::setw(20) << benchmark_results[i].avg_GOPS
                << std::setw(20) << benchmark_results[i].avg_thpt_GBPS
                << std::endl;
    }
    std::cout << std::endl;

    // log results
    result_log_file << "Kernel SpMSpV Benchmark, Target = " << target << std::endl;
    result_log_file << "Using: "<< xclbin << std::endl;
    result_log_file << "Verification result: " << (all_passed?"PASSED":"FAILED") << std::endl;
    result_log_file << "  Details:" << std::endl;
    result_log_file << "  "
                << std::setw(50) << "test case"
                << std::setw(20) << "semiring"
                << std::setw(20) << "vector sparsity"
                << std::setw(10) << "result"
                << std::setw(20) << "time(ms)"
                << std::setw(20) << "performance(GOPS)"
                << std::setw(20) << "throughput(GB/s)"
                << std::endl;
    // result_log_file << "  " << line(50 + 20 + 20 + 10 + 20 + 20 + 20) << std::endl;

    for (size_t i = 0; i < benchmark_results.size(); i++) {
        if (i % vector_sparsity_set.size() == 0) {
            result_log_file << "  " << line(50 + 20 + 20 + 10 + 20 + 20 + 20) << std::endl;
        }
        result_log_file << "  "
                << std::setw(50) << benchmark_results[i].info.name
                << std::setw(20) << benchmark_results[i].info.semiring_str
                << std::setw(20) << benchmark_results[i].info.spv_str
                << std::setw(10) << ((benchmark_results[i].avg_time_ms >= 0) ? "PASSED" : "FAILED")
                << std::setw(20) << benchmark_results[i].avg_time_ms
                << std::setw(20) << benchmark_results[i].avg_GOPS
                << std::setw(20) << benchmark_results[i].avg_thpt_GBPS
                << std::endl;
    }
    result_log_file << std::endl;

    return 0;
}

#pragma GCC diagnostic pop
