// goes into test_module_spmv_spmspv.cpp

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

// #include "graphlily/synthesizer/overlay_synthesizer.h"
#include "graphlily/synthesizer/split_kernel_synthesizer.h"

#include "graphlily/module/spmv_module.h"
#include "graphlily/module/spmspv_module.h"

#include <ap_fixed.h>
#include <gtest/gtest.h>

#include "graphlily/global.h"
#include "graphlily/io/data_loader.h"
#include "graphlily/io/data_formatter.h"
#include <iostream>

#include <thread>

const graphlily::SemiringType test_semiring[] = {
    graphlily::ArithmeticSemiring,
    // graphlily::LogicalSemiring,
    // graphlily::TropicalSemiring
};

const graphlily::MaskType test_mask_type[] = {
    graphlily::kNoMask,
    // graphlily::kMaskWriteToOne,
    // graphlily::kMaskWriteToZero
};

std::string target = "sw_emu";
std::string dataset_folder = "/work/shared/common/project_build/graphblas/"
                             "data/sparse_matrix_graph";
// std::string dataset_folder= "/path/to/data";
uint32_t spmv_out_buf_bank_size = 1024 * 8;
uint32_t spmv_vec_buf_bank_size = 1024 * 4;
uint32_t spmv_pe_num = graphlily::pack_size * graphlily::num_hbm_channels;
uint32_t spmv_out_buf_len = spmv_out_buf_bank_size * spmv_pe_num;
uint32_t spmv_vec_buf_len = spmv_vec_buf_bank_size * graphlily::pack_size;
uint32_t spmspv_out_buf_bank_size = 1024 * 2;
uint32_t spmspv_out_buf_len = spmspv_out_buf_bank_size * graphlily::pack_size;

void clean_proj_folder() {
    std::string command = "rm -rf ./" + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
}


template<typename data_t>
void verify(std::vector<float, aligned_allocator<float>> &reference_results,
            std::vector<data_t, aligned_allocator<data_t>> &kernel_results,
            std::vector<float, aligned_allocator<float>> &mask_vals) {
    ASSERT_EQ(reference_results.size(), kernel_results.size());
    float epsilon = 0.0001;
    for (size_t i = 0; i < reference_results.size(); i++) {
        ASSERT_TRUE(abs(kernel_results[i].to_float() - reference_results[i]) < epsilon)
            << kernel_results[i].to_float() << " output while "
            << reference_results[i] << " expected with "
            << mask_vals[i] << " for mask" << std::endl;
    }
}

std::string gen_test_case_name(graphlily::SemiringType semiring,
                                      graphlily::MaskType mask_type,
                                      std::string matrix_name) {
    std::string name = matrix_name + "_";
    switch (semiring.op) {
        case graphlily::kMulAdd:
            name += "Arithmetic_";
            break;
        case graphlily::kLogicalAndOr:
            name += "Logical_";
            break;
        case graphlily::kAddMin:
            name += "Tropical_";
            break;
        default:
            std::cout << "ERROR! Unsupported Semiring!" << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
    switch (mask_type) {
        case graphlily::kNoMask:
            name += "NoMask";
            break;
        case graphlily::kMaskWriteToOne:
            name += "WriteToOne";
            break;
        case graphlily::kMaskWriteToZero:
            name += "WriteToZero";
            break;
        default:
            std::cout << "ERROR! Unsupported Mask Type!" << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
    return name;
}

// for SpMV tests
std::string gen_test_case_name(graphlily::SemiringType semiring,
                                      graphlily::MaskType mask_type,
                                      std::string matrix_name,
                                      bool skip_empty_rows) {
    std::string suffix = skip_empty_rows ? "SkipEmptyRows" : "NotSkipRows";
    return gen_test_case_name(semiring, mask_type, matrix_name) + "_" + suffix;
}

// for SpMSpV tests
std::string gen_test_case_name(graphlily::SemiringType semiring,
                                      graphlily::MaskType mask_type,
                                      std::string matrix_name,
                                      float vector_sparsity) {
    float spv = vector_sparsity * 100;
    char c[20];
    sprintf(c, "%3.4f%%", spv);
    return gen_test_case_name(semiring, mask_type, matrix_name) + "_" + c;
}


TEST(Synthesize, NULL) {
    graphlily::synthesizer::SplitKernelSynthesizer synthesizer(graphlily::num_hbm_channels,
                                                           spmv_out_buf_len,
                                                           spmspv_out_buf_len,
                                                           spmv_vec_buf_len);
    synthesizer.set_target(target);
    synthesizer.synthesize();
}

void _test_spmv_module(graphlily::module::SpMVModule<graphlily::val_t, graphlily::val_t> &module,
                       graphlily::SemiringType semiring,
                       graphlily::MaskType mask_type,
                       std::string matrix_id,
                       CSRMatrix<float> const &csr_matrix,
                       bool skip_empty_rows) {
    std::cout << gen_test_case_name(semiring, mask_type, matrix_id, skip_empty_rows) << std::endl;
    using aligned_dense_vec_t = graphlily::aligned_dense_vec_t;
    using aligned_dense_float_vec_t = graphlily::aligned_dense_float_vec_t;

    module.set_semiring(semiring);
    module.set_mask_type(mask_type);

    size_t vector_length = csr_matrix.num_cols;

    aligned_dense_float_vec_t vector_float(vector_length);
    // std::generate(vector_float.begin(), vector_float.end(), []{return (float)(rand() % 10) / 10;});
    std::generate(vector_float.begin(), vector_float.end(), []{return 1.0;});
    aligned_dense_vec_t vector(vector_float.begin(), vector_float.end());

    aligned_dense_float_vec_t mask_float(vector_length);
    std::generate(mask_float.begin(), mask_float.end(), []{return (float)(rand() % 2);});
    aligned_dense_vec_t mask(mask_float.begin(), mask_float.end());

    module.load_and_format_matrix(csr_matrix, skip_empty_rows);

    aligned_dense_vec_t kernel_results;
    std::thread device_compute([&]{
        module.send_matrix_host_to_device();
        module.send_vector_host_to_device(vector);
        module.send_mask_host_to_device(mask);
        module.run();
        kernel_results = module.send_results_device_to_host();
    });

    aligned_dense_float_vec_t reference_results;
    if (mask_type == graphlily::kNoMask) {
        reference_results = module.compute_reference_results(vector_float);
    } else {
        reference_results = module.compute_reference_results(vector_float, mask_float);
    }

    device_compute.join();

    std::ofstream f("output.txt");
    
    for (size_t i = 0; i < reference_results.size(); i++) {
        float diff = (kernel_results[i].to_float() - reference_results[i]);
        if (abs(diff) > 0.0001) {
            f << "i: " << i << " ref result: " << reference_results[i] 
            << " kernel res: " << kernel_results[i]
            << " diff: " << diff << std::endl;
        }
    }

    f.close();

    verify<graphlily::val_t>(reference_results, kernel_results, mask_float);

}


std::vector<float> csr_matrix_vector_multiply(const CSRMatrix<float> &matrix, const std::vector<float> &vector) {
    std::vector<float> result(matrix.num_rows, 0.0);
    
    // std::cout << "Matrix dimensions: (" << matrix.num_rows << ", " << matrix.num_cols << ")" << std::endl;
    // std::cout << "Vector size: " << vector.size() << std::endl;

    for (size_t row = 0; row < matrix.num_rows; row++) {
        for (size_t idx = matrix.adj_indptr[row]; idx < matrix.adj_indptr[row + 1]; idx++) {
            size_t col = matrix.adj_indices[idx];
            
            if (col >= vector.size()) {
                std::cout << "Error: col index " << col << " out of bounds for vector of size " << vector.size() << std::endl;
            }
            
            result[row] += matrix.adj_data[idx] * vector[col];
        }
    }

    return result;
}


std::vector<float> csr_matrix_vector_multiply_combined(
    const CSRMatrix<float> &matrix,
    const std::vector<float> &vector,
    size_t input_matrix1_num_rows,
    size_t input_matrix1_start_col,
    size_t input_matrix1_end_col,
    size_t input_matrix2_start_col,
    size_t input_matrix2_end_col
) {
    std::vector<float> result(matrix.num_rows, 0.0);

    // Perform matrix 1 vector multiplication
    for (size_t row = 0; row < input_matrix1_num_rows; row++) {
        for (size_t idx = matrix.adj_indptr[row]; idx < matrix.adj_indptr[row + 1]; idx++) {
            size_t col = matrix.adj_indices[idx];

            if (col >= vector.size()) {
                std::cout << "Error: col index " << col << " out of bounds for vector of size " << vector.size() << std::endl;
            }

            result[row] += matrix.adj_data[idx] * vector[col];
        }
    }

    // Perform matrix 2 vector multiplication
    for (size_t row = input_matrix1_num_rows; row < matrix.num_rows; row++) {
        for (size_t idx = matrix.adj_indptr[row]; idx < matrix.adj_indptr[row + 1]; idx++) {
            size_t col = matrix.adj_indices[idx];

            if (col >= vector.size()) {
                std::cout << "Error: col index " << col << " out of bounds for vector of size " << vector.size() << std::endl;
            }

            result[row] += matrix.adj_data[idx] * vector[col];
        }
    }

    return result;
}



bool compare_vectors(const std::vector<float>& vec1, const std::vector<float>& vec2, float tolerance = 1e-6) {
    if (vec1.size() != vec2.size()) {
        return false;
    }

    for (size_t i = 0; i < vec1.size(); i++) {
        if (std::abs(vec1[i] - vec2[i]) > tolerance) {
            return false;
        }
    }

    return true;
}

void print_vector_portion(const std::vector<float>& vec, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}


void print_csr_matrix(const CSRMatrix<float>& matrix) {
    // Convert the CSR matrix to a dense 2D vector
    std::vector<std::vector<float>> dense_matrix(matrix.num_rows, std::vector<float>(matrix.num_cols, 0.0));

    for (size_t row = 0; row < matrix.num_rows; ++row) {
        for (size_t index = matrix.adj_indptr[row]; index < matrix.adj_indptr[row + 1]; ++index) {
            size_t col = matrix.adj_indices[index];
            float value = matrix.adj_data[index];
            dense_matrix[row][col] = value;
        }
    }

    // Print the dense matrix
    for (size_t row = 0; row < matrix.num_rows; ++row) {
        for (size_t col = 0; col < matrix.num_cols; ++col) {
            std::cout << dense_matrix[row][col] << " ";
        }
        std::cout << std::endl;
    }
}


void multitenancy_two_matrices_test(const std::string &data_path_1, const std::string &data_path_2) {
    // Load the first dense_32 matrix
    CSRMatrix<float> matrix_1 = graphlily::io::load_csr_matrix_from_float_npz(data_path_1);
    for (auto &x : matrix_1.adj_data) x = 1.0 / matrix_1.num_rows;

    // Load the second dense_32 matrix
    CSRMatrix<float> matrix_2 = graphlily::io::load_csr_matrix_from_float_npz(data_path_2);
    for (auto &x : matrix_2.adj_data) x = 1.0 / matrix_2.num_rows;

    // Combine the dense_32 matrices
    CSRMatrix<float> combined_matrix = graphlily::io::load_and_combine_csr_matrices_from_float_npz(matrix_1, matrix_2);

    graphlily::io::util_round_csr_matrix_dim(
        combined_matrix,
        graphlily::num_hbm_channels * graphlily::pack_size,
        graphlily::pack_size
    );

    // Create unity vectors for the original matrices and the combined matrix
    std::vector<float> matrix_1_vector(matrix_1.num_rows, 1.0);
    std::vector<float> matrix_2_vector(matrix_2.num_rows, 1.0);
    std::vector<float> combined_vector(combined_matrix.num_rows, 1.0);

    // Perform unity vector multiplication with the original matrices
    std::vector<float> original_matrix1_result = csr_matrix_vector_multiply(matrix_1, matrix_1_vector);
    std::vector<float> original_matrix2_result = csr_matrix_vector_multiply(matrix_2, matrix_2_vector);


    std::vector<float> combined_result = csr_matrix_vector_multiply_combined(
        combined_matrix,
        combined_vector,
        matrix_1.num_rows,
        0,
        matrix_1.num_cols,
        matrix_2.num_cols,
        combined_matrix.num_cols
    );


    // Split the result vector into parts corresponding to the original matrices
    std::vector<float> matrix1_result(combined_result.begin(), combined_result.begin() + matrix_1.num_rows);
    std::vector<float> matrix2_result(combined_result.begin() + matrix_1.num_rows, combined_result.begin() + matrix_1.num_rows + matrix_2.num_rows);

    bool dense_32_correct_1 = compare_vectors(original_matrix1_result, matrix1_result);
    bool dense_32_correct_2 = compare_vectors(original_matrix2_result, matrix2_result);

    if (!dense_32_correct_1) {
        std::cout << "Matrix 1 results:" << std::endl;
        std::cout << "Original Part 1: ";
        print_vector_portion(original_matrix1_result, 0, 32);
        std::cout << "Result Part 1: ";
        print_vector_portion(matrix1_result, 0, 32);
    }

    if (!dense_32_correct_2) {
        std::cout << "Matrix 2 results:" << std::endl;
        std::cout << "Original Part 2: ";
        print_vector_portion(original_matrix2_result, 0, 32);
        std::cout << "Result Part 2: ";
        print_vector_portion(matrix2_result, 0, 32);
    }

    // std::cout << "Combined matrix (top-left corner):" << std::endl;
    // print_csr_matrix(combined_matrix);

    if (dense_32_correct_1 && dense_32_correct_2) {
        std::cout << "The matrix-vector multiplication results are correct." << std::endl;
    } else {
        std::cout << "The matrix-vector multiplication results are incorrect." << std::endl;
    }
}





TEST(SpMV, MultipleCases) {
    graphlily::module::SpMVModule<graphlily::val_t, graphlily::val_t> module(graphlily::num_hbm_channels,
                                                                             spmv_out_buf_len,
                                                                             spmv_vec_buf_len);
    module.set_target(target);
    // module.set_up_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");
    module.set_up_split_kernel_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    // TODO: Combine these two matrices to make a 10,032 x 10,032 and use that matrix when pushing through
    // dense 32 x 32
    std::string csr_float_npz_path = dataset_folder + "/dense_32_csr_float32.npz";
    CSRMatrix<float> dense_32_matrix = graphlily::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
    graphlily::io::util_round_csr_matrix_dim( //! maybe this use for the matrix dimensions, row should be a multiple of 8*num_channels, column should be a multiple of 8
        dense_32_matrix,
        graphlily::num_hbm_channels * graphlily::pack_size,
        graphlily::pack_size);
    for (auto &x : dense_32_matrix.adj_data) x = 1.0 / dense_32_matrix.num_rows;

    // uniform (10K x 10K avg. degree 10)
    csr_float_npz_path = dataset_folder + "/uniform_10K_10_csr_float32.npz";
    CSRMatrix<float> uniform_10K_matrix = graphlily::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
    graphlily::io::util_round_csr_matrix_dim(
        uniform_10K_matrix,
        graphlily::num_hbm_channels * graphlily::pack_size,
        graphlily::pack_size);
    for (auto &x : uniform_10K_matrix.adj_data) x = 1.0 / uniform_10K_matrix.num_rows;


    // google plus (108K x 108K, 13M Nnz)
    csr_float_npz_path = dataset_folder + "/gplus_108K_13M_csr_float32.npz";
    CSRMatrix<float> gplus_108K_matrix = graphlily::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
    graphlily::io::util_round_csr_matrix_dim(
        gplus_108K_matrix,
        graphlily::num_hbm_channels * graphlily::pack_size,
        graphlily::pack_size);
    for (auto &x : gplus_108K_matrix.adj_data) x = 1.0 / gplus_108K_matrix.num_rows;

    std::map<std::string, CSRMatrix<float>> test_cases = {
        // { "dense32", dense_32_matrix },
        { "uniform10K10", uniform_10K_matrix },
        // { "google+", gplus_108K_matrix },
    };

    for (const auto &x : test_cases ) {
        for (const auto sr : test_semiring) {
            for (const auto msk_t : test_mask_type) {
                _test_spmv_module(module, sr, msk_t, x.first, x.second, false);
                _test_spmv_module(module, sr, msk_t, x.first, x.second, true);
            }
        }
    }
}

TEST(SpMV_Multitenancy, MultipleCases) {
    graphlily::module::SpMVModule<graphlily::val_t, graphlily::val_t> module(graphlily::num_hbm_channels,
                                                                             spmv_out_buf_len,
                                                                             spmv_vec_buf_len);
    module.set_target(target);
    // module.set_up_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");
    module.set_up_split_kernel_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    std::cout << "Combining 2 Dense_32 matrices together" << std::endl;

    // Load the first dense_32_csr_float32 matrix
    std::string csr_float_npz_path_1 = dataset_folder + "/dense_32_csr_float32.npz";

    // Load the second dense_32_csr_float32 matrix
    std::string csr_float_npz_path_2 = dataset_folder + "/dense_32_csr_float32.npz";

    multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    std::cout << "Combining 2 uniform_10K matrices together" << std::endl;

    // Load the first uniform_10K_10_csr_float32 matrix
    csr_float_npz_path_1 = dataset_folder + "/uniform_10K_10_csr_float32.npz";

    // Load the second uniform_10K_10_csr_float32 matrix
    csr_float_npz_path_2 = dataset_folder + "/uniform_10K_10_csr_float32.npz";

    std::cout << "Combining 2 gplus_108K_13M_csr_float32 matrices together" << std::endl;

    // Load the first google plus (108K x 108K, 13M Nnz)
    csr_float_npz_path_1 = dataset_folder + "/gplus_108K_13M_csr_float32.npz";

    // Load the second google plus (108K x 108K, 13M Nnz)
    csr_float_npz_path_2 = dataset_folder + "/gplus_108K_13M_csr_float32.npz";

    multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    std::cout << "Combining Dense_32 and uniform_10K matrices together (in that order)" << std::endl;

    // Load the dense_32 matrix
    csr_float_npz_path_1 = dataset_folder + "/dense_32_csr_float32.npz";

    // Load the uniform_10K_10_csr_float32 matrix
    csr_float_npz_path_2 = dataset_folder + "/uniform_10K_10_csr_float32.npz";

    multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    std::cout << "Combining uniform_10K and Dense_32 matrices together (in that order)" << std::endl;

    // Load the uniform_10K_10_csr_float32 matrix
    csr_float_npz_path_1 = dataset_folder + "/uniform_10K_10_csr_float32.npz";

    // Load the dense_32 matrix
    csr_float_npz_path_2 = dataset_folder + "/dense_32_csr_float32.npz";

    multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    std::cout << "Combining gplus_108K_13M_csr_float32 and dense_32_csr_float32 matrices together (in that order)" << std::endl;

    // Load the gplus_108K_13M_csr_float32 matrix
    csr_float_npz_path_1 = dataset_folder + "/gplus_108K_13M_csr_float32.npz";

    // Load the dense_32 matrix
    csr_float_npz_path_2 = dataset_folder + "/dense_32_csr_float32.npz";

    multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    std::cout << "Combining dense_32_csr_float32 and gplus_108K_13M_csr_float32 together (in that order)" << std::endl;

    // Load the dense_32 matrix
    csr_float_npz_path_1 = dataset_folder + "/dense_32_csr_float32.npz";

    // Load the gplus_108K_13M_csr_float32 matrix
    csr_float_npz_path_2 = dataset_folder + "/gplus_108K_13M_csr_float32.npz";

    multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    std::cout << "Combining gplus_108K_13M_csr_float32 and uniform_10K_10_csr_float32 matrices together (in that order)" << std::endl;

    // Load the gplus_108K_13M_csr_float32 matrix
    csr_float_npz_path_1 = dataset_folder + "/gplus_108K_13M_csr_float32.npz";

    // Load the uniform_10K_10_csr_float32 matrix
    csr_float_npz_path_2 = dataset_folder + "/uniform_10K_10_csr_float32.npz";

    multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    std::cout << "Combining uniform_10K_10_csr_float32 and gplus_108K_13M_csr_float32 together (in that order)" << std::endl;

    // Load the uniform_10K_10_csr_float32 matrix
    csr_float_npz_path_1 = dataset_folder + "/uniform_10K_10_csr_float32.npz";

    // Load the gplus_108K_13M_csr_float32 matrix
    csr_float_npz_path_2 = dataset_folder + "/gplus_108K_13M_csr_float32.npz";

    multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    // =========================================================================
    // Custom CSR Matrices
    // =========================================================================
    // std::cout << "Combining 2 eye_10_csr_float32 matrices together" << std::endl;

    // // personally created CSR files
    // std::string dataset_folder_1 = "/home/mah426/GraphLily/tests/test_data";
    // // Load the first eye_10_csr_float32 matrix
    // csr_float_npz_path_1 = dataset_folder_1 + "/eye_10_csr_float32.npz";

    // // Load the second eye_10_csr_float32 matrix
    // csr_float_npz_path_2 = dataset_folder_1 + "/eye_10_csr_float32.npz";
    // multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    // std::cout << "Combining 2 LabelP1 matrices together" << std::endl;
    
    // // Load the first LabelP1 matrix
    // csr_float_npz_path_1 = dataset_folder_1 + "/LabelP1.npz";
    // // Load the second LabelP1 matrix
    // csr_float_npz_path_2 = dataset_folder_1 + "/LabelP1.npz";
    // multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    // std::cout << "Combining 2 line_8_csr_float32 matrices together" << std::endl;

    // // Load the first line_8_csr_float32 matrix
    // csr_float_npz_path_1 = dataset_folder_1 + "/line_8_csr_float32.npz";
    // // Load the second line_8_csr_float32 matrix
    // csr_float_npz_path_2 = dataset_folder_1 + "/line_8_csr_float32.npz";
    // multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    // std::cout << "Combining 2 SimpleTest1 matrices together" << std::endl;

    // // Load the first SimpleTest1 matrix
    // csr_float_npz_path_1 = dataset_folder_1 + "/SimpleTest1.npz";
    // // Load the second SimpleTest1 matrix
    // csr_float_npz_path_2 = dataset_folder_1 + "/SimpleTest1.npz";
    // multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    // std::cout << "Combining eye_10_csr_float32 and LabelP1 matrices together" << std::endl;

    // // Load the eye_10_csr_float32 matrix
    // csr_float_npz_path_1 = dataset_folder_1 + "/eye_10_csr_float32.npz";

    // // Load the LabelP1 matrix
    // csr_float_npz_path_2 = dataset_folder_1 + "/LabelP1.npz";
    // multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    // std::cout << "Combining eye_10_csr_float32 and line_8_csr_float32 matrices together" << std::endl;

    // // Load the eye_10_csr_float32 matrix
    // csr_float_npz_path_1 = dataset_folder_1 + "/eye_10_csr_float32.npz";

    // // Load the line_8_csr_float32 matrix
    // csr_float_npz_path_2 = dataset_folder_1 + "/line_8_csr_float32.npz";
    // multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

    // std::cout << "Combining eye_10_csr_float32 and SimpleTest1 matrices together" << std::endl;

    // // Load the eye_10_csr_float32 matrix
    // csr_float_npz_path_1 = dataset_folder_1 + "/eye_10_csr_float32.npz";

    // // Load the SimpleTest1 matrix
    // csr_float_npz_path_2 = dataset_folder_1 + "/SimpleTest1.npz";
    // multitenancy_two_matrices_test( csr_float_npz_path_1, csr_float_npz_path_2);

}




// void _test_spmspv_module(graphlily::module::SpMSpVModule<graphlily::val_t,
//                                                          graphlily::val_t,
//                                                          graphlily::idx_val_t> &module,
//                          graphlily::SemiringType semiring,
//                          graphlily::MaskType mask_type,
//                          std::string matrix_id,
//                          CSCMatrix<float> const &csc_matrix,
//                          float vector_sparsity) {
//     std::cout << gen_test_case_name(semiring, mask_type, matrix_id, vector_sparsity) << std::endl;
//     using aligned_dense_vec_t = graphlily::aligned_dense_vec_t;
//     using aligned_sparse_vec_t = graphlily::aligned_sparse_vec_t;

//     module.set_semiring(semiring);
//     module.set_mask_type(mask_type);

//     // generate vector
//     unsigned vector_length = csc_matrix.num_cols;
//     unsigned vector_nnz_cnt = (unsigned)floor(vector_length * (1 - vector_sparsity));
//     unsigned vector_indices_increment = vector_length / vector_nnz_cnt;

//     graphlily::aligned_sparse_float_vec_t vector_float(vector_nnz_cnt);
//     for (size_t i = 0; i < vector_nnz_cnt; i++) {
//         vector_float[i].val = (float)(rand() % 10) / 10;
//         vector_float[i].index = i * vector_indices_increment;
//     }
//     graphlily::idx_float_t vector_head;
//     vector_head.index = vector_nnz_cnt;
//     vector_head.val = 0;
//     vector_float.insert(vector_float.begin(), vector_head);
//     aligned_sparse_vec_t vector(vector_float.size());
//     for (size_t i = 0; i < vector[0].index + 1; i++) {
//         vector[i].index = vector_float[i].index;
//         vector[i].val = vector_float[i].val;
//     }

//     // generate mask
//     unsigned mask_length = csc_matrix.num_rows;
//     graphlily::aligned_dense_float_vec_t mask_float(mask_length);
//     std::generate(mask_float.begin(), mask_float.end(), []{return (float)(rand() % 2);});
//     aligned_dense_vec_t mask(mask_float.begin(), mask_float.end());

//     module.load_and_format_matrix(csc_matrix);

//     // run the kernel
//     aligned_dense_vec_t kernel_results_dense;
//     std::thread device_compute([&]{
//         module.send_matrix_host_to_device();
//         module.send_mask_host_to_device(mask);
//         module.send_vector_host_to_device(vector);
//         module.run();
//         auto sparse_vec = module.send_results_device_to_host();
//         kernel_results_dense = graphlily::convert_sparse_vec_to_dense_vec<aligned_sparse_vec_t,
//             aligned_dense_vec_t, graphlily::val_t>(sparse_vec, vector_length, semiring.zero);
//     });

//     graphlily::aligned_dense_float_vec_t reference_results =
//         module.compute_reference_results(vector_float, mask_float);

//     device_compute.join();
//     verify<graphlily::val_t>(reference_results, kernel_results_dense, mask_float);
// }


// TEST(SpMSpV, MultipleCases) {
//     graphlily::module::SpMSpVModule<graphlily::val_t,
//                                     graphlily::val_t,
//                                     graphlily::idx_val_t> module(spmspv_out_buf_len);
//     module.set_target(target);
//     module.set_up_split_kernel_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

//     // dense 1K x 1K
//     CSCMatrix<float> csc_matrix_dense1K = graphlily::io::csr2csc(
//         graphlily::io::load_csr_matrix_from_float_npz(dataset_folder + "/dense_1K_csr_float32.npz"));
//     for (auto &x : csc_matrix_dense1K.adj_data) x = 1.0 / csc_matrix_dense1K.num_rows;

//     // uniform (10K x 10K avg. degree 10)
//     CSCMatrix<float> csc_matrix_uniform10K10 = graphlily::io::csr2csc(
//         graphlily::io::load_csr_matrix_from_float_npz(dataset_folder + "/uniform_10K_10_csr_float32.npz"));
//     for (auto &x : csc_matrix_uniform10K10.adj_data) x = 1.0 / csc_matrix_uniform10K10.num_rows;

//     // google plus (108K x 108K, 13M Nnz)
//     CSCMatrix<float> csc_matrix_gpuls = graphlily::io::csr2csc(
//         graphlily::io::load_csr_matrix_from_float_npz(dataset_folder + "/gplus_108K_13M_csr_float32.npz"));
//     for (auto &x : csc_matrix_gpuls.adj_data) x = 1.0 / csc_matrix_gpuls.num_rows;

//     // bank conflict test case
//     CSCMatrix<float> csc_matrix_conflict;
//     unsigned conflict_matirx_size = 1024;
//     csc_matrix_conflict.num_rows = conflict_matirx_size;
//     csc_matrix_conflict.num_cols = conflict_matirx_size;
//     csc_matrix_conflict.adj_data.resize(conflict_matirx_size/8*conflict_matirx_size);
//     for (auto &x : csc_matrix_conflict.adj_data) x = 1.0 / csc_matrix_conflict.num_rows;
//     csc_matrix_conflict.adj_indices.resize(conflict_matirx_size/8*conflict_matirx_size);
//     for (size_t i = 0; i < conflict_matirx_size; i++) {
//         for (size_t j = 0; j < conflict_matirx_size/8; j++) {
//             csc_matrix_conflict.adj_indices[i * (conflict_matirx_size/8) + j] = j * 8 + i % 8;
//         }
//     }
//     csc_matrix_conflict.adj_indptr.resize(conflict_matirx_size + 1);
//     for (size_t i = 0; i < conflict_matirx_size + 1; i++) {
//         csc_matrix_conflict.adj_indptr[i] = i * (conflict_matirx_size / 8);
//     }

//     _test_spmspv_module(module, graphlily::ArithmeticSemiring, graphlily::kNoMask,
//         "conflict" + std::to_string(conflict_matirx_size), csc_matrix_conflict, 0.00);

//     std::map<std::string, CSCMatrix<float>> test_cases = {
//         { "dense1K", csc_matrix_dense1K },
//         { "uniform10K10", csc_matrix_uniform10K10 },
//         // { "google+", csc_matrix_gpuls },
//     };

//     for (const auto &x : test_cases ) {
//         for (const auto sr : test_semiring) {
//             for (const auto msk_t : test_mask_type) {
//                 _test_spmspv_module(module, sr, msk_t, x.first, x.second, 0.50);
//                 _test_spmspv_module(module, sr, msk_t, x.first, x.second, 0.99);
//             }
//         }
//     }
// }


TEST(Clean, NULL) {
    clean_proj_folder();
}


int main(int argc, char ** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#pragma GCC diagnostic pop
