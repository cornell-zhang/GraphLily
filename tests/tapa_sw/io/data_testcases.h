#ifndef _IO_DATA_TESTCASES_H_
#define _IO_DATA_TESTCASES_H_

#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>

#include "data_loader.h"
#include "data_formatter.h"

using namespace spmv_::io;
using namespace spmspv_::io;

using spmv_::io::load_csr_matrix_from_float_npz;

using spmspv_::io::csr2csc;
using spmspv_::io::CSCMatrix;

//---------------------------------------------------------------
// test case utils
//---------------------------------------------------------------

CSRMatrix<float> create_dense_CSR (
    unsigned num_rows,
    unsigned num_cols
) {
    CSRMatrix<float> mat_f;
    mat_f.num_rows = num_rows;
    mat_f.num_cols = num_cols;
    mat_f.adj_data.resize(num_rows * num_cols);
    mat_f.adj_indices.resize(num_rows * num_cols);
    mat_f.adj_indptr.resize(num_rows + 1);

    for (auto &x : mat_f.adj_data) {x = 1;}

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            mat_f.adj_indices[i*num_cols + j] = j;
        }
    }
    for (size_t i = 0; i < num_rows + 1; i++) {
        mat_f.adj_indptr[i] = num_cols*i;
    }
    return mat_f;
}

CSRMatrix<float> create_uniform_sparse_CSR (
    unsigned num_rows,
    unsigned num_cols,
    unsigned nnz_per_row
) {
    CSRMatrix<float> mat_f;
    mat_f.num_rows = num_rows;
    mat_f.num_cols = num_cols;
    mat_f.adj_data.resize(num_rows * nnz_per_row);
    mat_f.adj_indices.resize(num_rows * nnz_per_row);
    mat_f.adj_indptr.resize(num_rows + 1);

    for (auto &x : mat_f.adj_data) {x = 1;}

    unsigned indice_step = num_cols / nnz_per_row;
    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < nnz_per_row; j++) {
            mat_f.adj_indices[i*nnz_per_row + j] = (indice_step*j + i) % num_cols;
        }
    }
    for (size_t i = 0; i < num_rows + 1; i++) {
        mat_f.adj_indptr[i] = nnz_per_row*i;
    }
    return mat_f;
}

//---------------------------------------------------------------
// test cases
//---------------------------------------------------------------

std::string graph_dataset_dir();
std::string nn_dataset_dir();

#ifdef TEST_SPMV

bool spmv_test_harness (
    std::string bitstream,
    spmv_::io::CSRMatrix<float> &ext_matrix,
    bool skip_empty_rows
);

bool spmv_test_basic(std::string bitstream) {
    std::cout << "------ Running test: on basic dense matrix " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f = create_dense_CSR(128, 128);
    for (auto &x : mat_f.adj_data) {x = 1;}
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmv_test_basic_sparse(std::string bitstream) {
    std::cout << "------ Running test: on basic sparse matrix " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f = create_uniform_sparse_CSR(1000, 1024, 10);
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmv_test_medium_sparse(std::string bitstream) {
    std::cout << "------ Running test: on uniform 10K 10 (100K, 1M) " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f = create_uniform_sparse_CSR(10000, 10000, 10);
    for (auto &x : mat_f.adj_data) {x = 1;}
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmv_test_gplus(std::string bitstream) {
    std::cout << "------ Running test: on google_plus (108K, 13M) " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f =
        spmv_::io::load_csr_matrix_from_float_npz(graph_dataset_dir() + "gplus_108K_13M_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmv_test_ogbl_ppa(std::string bitstream) {
    std::cout << "------ Running test: on ogbl_ppa (576K, 42M) " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f =
        spmv_::io::load_csr_matrix_from_float_npz(graph_dataset_dir() + "ogbl_ppa_576K_42M_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmv_test_hollywood(std::string bitstream) {
    std::cout << "------ Running test: on hollywood (1M, 113M) " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f =
        spmv_::io::load_csr_matrix_from_float_npz(graph_dataset_dir() + "hollywood_1M_113M_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmv_test_pokec(std::string bitstream) {
    std::cout << "------ Running test: on pokec (1633K, 31M) " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f =
        spmv_::io::load_csr_matrix_from_float_npz(graph_dataset_dir() + "pokec_1633K_31M_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmv_test_ogbn_products(std::string bitstream) {
    std::cout << "------ Running test: on ogbn_products (2M, 124M) " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f =
        spmv_::io::load_csr_matrix_from_float_npz(graph_dataset_dir() + "ogbn_products_2M_124M_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmv_test_orkut(std::string bitstream) {
    std::cout << "------ Running test: on orkut (3M, 213M) " << std::endl;
    spmv_::io::CSRMatrix<float> mat_f =
        spmv_::io::load_csr_matrix_from_float_npz(graph_dataset_dir() + "orkut_3M_213M_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(bitstream, mat_f, false)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmv_test_transformer_50_t(std::string bitstream) {
    std::cout << "------ Running test: on transformer-50-t" << std::endl;
    spmv_::io::CSRMatrix<float> mat_f =
        spmv_::io::load_csr_matrix_from_float_npz(nn_dataset_dir() + "transformer_50_512_33288_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(bitstream, mat_f, true)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmv_test_transformer_95_t(std::string bitstream) {
    std::cout << "------ Running test: on transformer-95-t" << std::endl;
    spmv_::io::CSRMatrix<float> mat_f =
        spmv_::io::load_csr_matrix_from_float_npz(nn_dataset_dir() + "transformer_95_512_33288_csr_float32.npz");
    for (auto &x : mat_f.adj_data) {x = 1 / mat_f.num_cols;}
    if (spmv_test_harness(bitstream, mat_f, true)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

#elif TEST_SPMSPV

bool spmspv_test_harness (
    std::string bitstream,
    CSCMatrix<float> &csc_matrix_float,
    float vector_sparsity
);

bool spmspv_test_dense32(std::string bitstream) {
    std::cout << "------ Running test: on basic dense matrix " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(create_dense_CSR(32, 32));
    for (auto &x : mat_f.adj_data) x = 1.0;
    if (spmspv_test_harness(bitstream, mat_f, 0.0)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmspv_test_basic(std::string bitstream) {
    std::cout << "------ Running test: on basic dense matrix " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(create_dense_CSR(128, 128));
    for (auto &x : mat_f.adj_data) x = 1.0;
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmspv_test_basic_sparse(std::string bitstream) {
    std::cout << "------ Running test: on basic sparse matrix " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(create_uniform_sparse_CSR(1000, 1024, 10));
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmspv_test_medium_sparse(std::string bitstream) {
    std::cout << "------ Running test: on uniform 10K 10 (100K, 1M) " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(create_uniform_sparse_CSR(10000, 10000, 10));
    for (auto &x : mat_f.adj_data) x = 1.0;
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmspv_test_gplus(std::string bitstream) {
    std::cout << "------ Running test: on google_plus (108K, 13M) " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(
        load_csr_matrix_from_float_npz(graph_dataset_dir() + "gplus_108K_13M_csr_float32.npz"));
    for (auto &x : mat_f.adj_data) x = 1.0 / mat_f.num_cols;
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmspv_test_ogbl_ppa(std::string bitstream) {
    std::cout << "------ Running test: on ogbl_ppa (576K, 42M) " << std::endl;
    CSCMatrix<float> mat_f = csr2csc(
        load_csr_matrix_from_float_npz(graph_dataset_dir() + "ogbl_ppa_576K_42M_csr_float32.npz"));
    for (auto &x : mat_f.adj_data) x = 1.0 / mat_f.num_cols;
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmspv_test_transformer_50_t(std::string bitstream) {
    std::cout << "------ Running test: on transformer-50-t" << std::endl;
    CSCMatrix<float> mat_f = csr2csc(
        load_csr_matrix_from_float_npz(nn_dataset_dir() + "transformer_50_512_33288_csr_float32.npz"));
    for (auto &x : mat_f.adj_data) x = 1.0 / mat_f.num_cols;
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

bool spmspv_test_transformer_95_t(std::string bitstream) {
    std::cout << "------ Running test: on transformer-95-t" << std::endl;
    CSCMatrix<float> mat_f = csr2csc(
        load_csr_matrix_from_float_npz(nn_dataset_dir() + "transformer_95_512_33288_csr_float32.npz"));
    for (auto &x : mat_f.adj_data) x = 1.0 / mat_f.num_cols;
    if (spmspv_test_harness(bitstream, mat_f, 0.5)) {
        std::cout << "INFO : Testcase passed." << std::endl;
        return true;
    } else {
        std::cout << "INFO : Testcase failed." << std::endl;
        return false;
    }
}

#endif

#endif
