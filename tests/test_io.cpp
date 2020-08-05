#include <cstdlib>
#include <iostream>
#include <limits>
#include "graphblas/io/data_loader.h"
#include "graphblas/io/data_formatter.h"

using namespace graphblas::io;


template<typename T>
void check_vector_equal(std::vector<T> const& vec1, std::vector<T> const& vec2) {
    if (!(vec1.size() == vec2.size())) {
        std::cout << "Size mismatch!" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (!(std::equal(vec1.begin(), vec1.end(), vec2.begin()))) {
        std::cout << "Value mismatch!" << std::endl;
        exit(EXIT_FAILURE);
    }
}


template<typename T, uint32_t pack_size>
void check_packed_vector_equal(std::vector<T> const& vec1, std::vector<T> const& vec2) {
    if (!(vec1.size() == vec2.size())) {
        std::cout << "Size mismatch!" << std::endl;
        exit(EXIT_FAILURE);
    }
    auto predicate = [](T a, T b) {
        for (uint32_t i = 0; i < pack_size; i++) {
            if (a.data[i] != b.data[i]) {
                return false;
            }
        }
        return true;
    };
    if (!(std::equal(vec1.begin(), vec1.end(), vec2.begin(), predicate))) {
        std::cout << "Value mismatch!" << std::endl;
        exit(EXIT_FAILURE);
    }
}


void test_create_csr_matrix() {
    uint32_t num_rows = 5;
    uint32_t num_cols = 5;
    std::vector<float> adj_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<uint32_t> adj_indices = {0, 1, 2, 3, 0, 2, 1, 3, 2};
    std::vector<uint32_t> adj_indptr = {0, 4, 6, 7, 8, 9};

    CSRMatrix<float> csr_matrix = create_csr_matrix(num_rows, num_cols, adj_data, adj_indices, adj_indptr);

    assert(csr_matrix.num_rows == num_rows);
    assert(csr_matrix.num_cols == num_cols);
    check_vector_equal<float>(csr_matrix.adj_data, adj_data);
    check_vector_equal<uint32_t>(csr_matrix.adj_indices, adj_indices);
    check_vector_equal<uint32_t>(csr_matrix.adj_indptr, adj_indptr);

    std::cout << "test_create_csr_matrix passed" << std::endl;
}


void test_load_csr_matrix_from_float_npz() {
    CSRMatrix<float> csr_matrix = load_csr_matrix_from_float_npz("../test_data/eye_10_csr_float32.npz");

    assert(csr_matrix.num_rows == 10);
    assert(csr_matrix.num_cols == 10);
    check_vector_equal<float>(csr_matrix.adj_data, std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    check_vector_equal<uint32_t>(csr_matrix.adj_indices, std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    check_vector_equal<uint32_t>(csr_matrix.adj_indptr, std::vector<uint32_t>{0,1,2,3, 4, 5, 6, 7, 8, 9, 10});

    std::cout << "test_load_csr_matrix_from_float_npz passed" << std::endl;
}


void test_csr_matrix_convert_from_float() {
    CSRMatrix<float> csr_matrix_float = load_csr_matrix_from_float_npz("../test_data/eye_10_csr_float32.npz");
    CSRMatrix<int> csr_matrix_int = csr_matrix_convert_from_float<int>(csr_matrix_float);

    assert(csr_matrix_int.num_rows == 10);
    assert(csr_matrix_int.num_cols == 10);
    check_vector_equal<int>(csr_matrix_int.adj_data, std::vector<int>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    check_vector_equal<uint32_t>(csr_matrix_int.adj_indices, std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    check_vector_equal<uint32_t>(csr_matrix_int.adj_indptr, std::vector<uint32_t>{0,1,2,3, 4, 5, 6, 7, 8, 9, 10});

    std::cout << "test_csr_matrix_convert_from_float passed" << std::endl;

}


// csr_matrix_1 is:
//     [[1, 2, 3, 4],
//      [5, 0, 6, 0],
//      [0, 7, 0, 0],
//      [0, 0, 0, 8]]
const static CSRMatrix<float> csr_matrix_1 = {
    .num_rows=4,
    .num_cols=4,
    .adj_data=std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8},
    .adj_indices=std::vector<uint32_t>{0, 1, 2, 3, 0, 2, 1, 3},
    .adj_indptr=std::vector<uint32_t>{0, 4, 6, 7, 8},
};


// csr_matrix_2 is:
//     [[1, 2, 3, 4, 1, 2, 3, 4],
//      [5, 0, 6, 0, 5, 0, 6, 0],
//      [0, 7, 0, 0, 0, 7, 0, 0],
//      [0, 0, 0, 8, 0, 0, 0, 8]
const static CSRMatrix<float> csr_matrix_2 = {
    .num_rows=4,
    .num_cols=8,
    .adj_data=std::vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 5, 6, 7, 7, 8, 8},
    .adj_indices=std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 1, 5, 3, 7},
    .adj_indptr=std::vector<uint32_t>{0, 8, 12, 14, 16},
};


void test_util_round_csr_matrix_dim() {
    CSRMatrix<float> M = csr_matrix_1;
    assert(M.num_rows == 4);
    assert(M.num_cols == 4);

    uint32_t row_divisor = 3;
    uint32_t col_divisor = 5;
    util_round_csr_matrix_dim(M, row_divisor, col_divisor);
    assert(M.num_rows == 6);
    assert(M.num_cols == 5);

    std::cout << "test_util_round_csr_matrix_dim passed" << std::endl;
}


void test_util_convert_csr_to_dds() {
    CSRMatrix<float> M = csr_matrix_1;
    uint32_t num_cols_per_partition = 3;
    uint32_t num_col_partitions = (M.num_cols + num_cols_per_partition - 1) / num_cols_per_partition;
    std::vector<float> partitioned_data[num_col_partitions];
    std::vector<uint32_t> partitioned_indices[num_col_partitions];
    std::vector<uint32_t> partitioned_indptr[num_col_partitions];

    util_convert_csr_to_dds<float>(M.num_rows,
                                   M.num_cols,
                                   M.adj_data.data(),
                                   M.adj_indices.data(),
                                   M.adj_indptr.data(),
                                   num_cols_per_partition,
                                   partitioned_data,
                                   partitioned_indices,
                                   partitioned_indptr);

    std::vector<float> reference_partition_1_data = {1, 2, 3, 5, 6, 7};
    std::vector<float> reference_partition_2_data = {4, 8};
    std::vector<uint32_t> reference_partition_1_indices = {0, 1, 2, 0, 2, 1};
    std::vector<uint32_t> reference_partition_2_indices = {0, 0};
    std::vector<uint32_t> reference_partition_1_indptr = {0, 3, 5, 6, 6};
    std::vector<uint32_t> reference_partition_2_indptr = {0, 1, 1, 1, 2};

    check_vector_equal<float>(partitioned_data[0], reference_partition_1_data);
    check_vector_equal<float>(partitioned_data[1], reference_partition_2_data);
    check_vector_equal<uint32_t>(partitioned_indices[0], reference_partition_1_indices);
    check_vector_equal<uint32_t>(partitioned_indices[1], reference_partition_2_indices);
    check_vector_equal<uint32_t>(partitioned_indptr[0], reference_partition_1_indptr);
    check_vector_equal<uint32_t>(partitioned_indptr[1], reference_partition_2_indptr);

    std::cout << "test_util_convert_csr_to_dds passed" << std::endl;
}


void test_util_reorder_rows_ascending_nnz() {
    CSRMatrix<float> M = csr_matrix_1;
    std::vector<float> reordered_data;
    std::vector<uint32_t> reordered_indices;
    std::vector<uint32_t> reordered_indptr;

    util_reorder_rows_ascending_nnz<float>(M.adj_data,
                                           M.adj_indices,
                                           M.adj_indptr,
                                           reordered_data,
                                           reordered_indices,
                                           reordered_indptr);

    // After reordering, the sparse matrix is:
    //     [[0, 7, 0, 0],
    //      [0, 0, 0, 8],
    //      [5, 0, 6, 0],
    //      [1, 2, 3, 4]]

    std::vector<float> reference_reordered_data = {7, 8, 5, 6, 1, 2, 3, 4};
    std::vector<uint32_t> reference_reordered_indices = {1, 3, 0, 2, 0, 1, 2, 3};
    std::vector<uint32_t> reference_reordered_indptr = {0, 1, 2, 4, 8};

    check_vector_equal<float>(reordered_data, reference_reordered_data);
    check_vector_equal<uint32_t>(reordered_indices, reference_reordered_indices);
    check_vector_equal<uint32_t>(reordered_indptr, reference_reordered_indptr);

    std::cout << "test_util_reorder_rows_ascending_nnz passed" << std::endl;
}


void test_util_pack_rows() {
    CSRMatrix<float> M = csr_matrix_1;
    const uint32_t num_hbm_channels = 2;
    const uint32_t num_PEs_per_hbm_channel = 2;
    typedef struct packed_data_type_ {float data[num_PEs_per_hbm_channel];} packed_data_type;
    typedef struct packed_index_type_ {uint32_t data[num_PEs_per_hbm_channel];} packed_index_type;

    std::vector<packed_data_type> packed_data[num_hbm_channels];
    std::vector<packed_index_type> packed_indices[num_hbm_channels];
    std::vector<packed_index_type> packed_indptr[num_hbm_channels];

    util_pack_rows<float, packed_data_type, packed_index_type>(M.adj_data,
                                                               M.adj_indices,
                                                               M.adj_indptr,
                                                               num_hbm_channels,
                                                               num_PEs_per_hbm_channel,
                                                               packed_data,
                                                               packed_indices,
                                                               packed_indptr);

    std::vector<packed_data_type> reference_packed_data_channel_1 = {{1,5}, {2,6}, {3,0}, {4,0}};
    std::vector<packed_index_type> reference_packed_indices_channel_1 = {{0,0}, {1,2}, {2,0}, {3,0}};
    std::vector<packed_index_type> reference_packed_indptr_channel_1 = {{0,0}, {4,2}};
    std::vector<packed_data_type> reference_packed_data_channel_2 = {{7,8}};
    std::vector<packed_index_type> reference_packed_indices_channel_2 = {{1,3}};
    std::vector<packed_index_type> reference_packed_indptr_channel_2 = {{0,0}, {1,1}};

    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        packed_data[0], reference_packed_data_channel_1);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        packed_indices[0], reference_packed_indices_channel_1);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        packed_indptr[0], reference_packed_indptr_channel_1);
    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        packed_data[1], reference_packed_data_channel_2);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        packed_indices[1], reference_packed_indices_channel_2);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        packed_indptr[1], reference_packed_indptr_channel_2);

    std::cout << "test_util_pack_rows passed" << std::endl;
}


void test_spmv_data_formatter_col_partitioning() {
    CSRMatrix<float> M = csr_matrix_2;
    const uint32_t out_buffer_len = 4;
    const uint32_t vector_buffer_len = 4;
    const uint32_t num_hbm_channels = 2;
    const uint32_t num_PEs_per_hbm_channel = 2;
    typedef struct packed_data_type_ {float data[num_PEs_per_hbm_channel];} packed_data_type;
    typedef struct packed_index_type_ {uint32_t data[num_PEs_per_hbm_channel];} packed_index_type;
    float val_marker = std::numeric_limits<float>::infinity();
    unsigned int idx_marker = std::numeric_limits<unsigned int>::max();

    SpMVDataFormatter<float, num_PEs_per_hbm_channel, packed_data_type, packed_index_type> formatter(M);
    formatter.format_pad_marker_end_of_row(out_buffer_len,
                                           vector_buffer_len,
                                           num_hbm_channels,
                                           val_marker,
                                           idx_marker);

    std::vector<packed_data_type> reference_data_col_partition_1_channel_1 =
        {{1,5}, {2,6}, {3,val_marker}, {4,0}, {val_marker,0}};
    std::vector<packed_index_type> reference_indices_col_partition_1_channel_1 =
        {{0,0}, {1,2}, {2,idx_marker}, {3,0}, {idx_marker,0}};
    std::vector<packed_data_type> reference_data_col_partition_1_channel_2 =
        {{7,8}, {val_marker,val_marker}};
    std::vector<packed_index_type> reference_indices_col_partition_1_channel_2 =
        {{1,3}, {idx_marker,idx_marker}};
    auto reference_data_col_partition_2_channel_1 = reference_data_col_partition_1_channel_1;
    auto reference_indices_col_partition_2_channel_1 = reference_indices_col_partition_1_channel_1;
    auto reference_data_col_partition_2_channel_2 = reference_data_col_partition_1_channel_2;
    auto reference_indices_col_partition_2_channel_2 = reference_indices_col_partition_1_channel_2;

    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        formatter.get_packed_data(0, 0, 0), reference_data_col_partition_1_channel_1);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        formatter.get_packed_indices(0, 0, 0), reference_indices_col_partition_1_channel_1);
    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        formatter.get_packed_data(0, 0, 1), reference_data_col_partition_1_channel_2);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        formatter.get_packed_indices(0, 0, 1), reference_indices_col_partition_1_channel_2);
    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        formatter.get_packed_data(0, 1, 0), reference_data_col_partition_2_channel_1);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        formatter.get_packed_indices(0, 1, 0), reference_indices_col_partition_2_channel_1);
    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        formatter.get_packed_data(0, 1, 1), reference_data_col_partition_2_channel_2);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        formatter.get_packed_indices(0, 1, 1), reference_indices_col_partition_2_channel_2);

    std::cout << "test_spmv_data_formatter_col_partitioning passed" << std::endl;
}


void test_spmv_data_formatter_row_partitioning() {
    CSRMatrix<float> M = csr_matrix_1;
    const uint32_t out_buffer_len = 2;
    const uint32_t vector_buffer_len = 4;
    const uint32_t num_hbm_channels = 1;
    const uint32_t num_PEs_per_hbm_channel = 2;
    typedef struct packed_data_type_ {float data[num_PEs_per_hbm_channel];} packed_data_type;
    typedef struct packed_index_type_ {uint32_t data[num_PEs_per_hbm_channel];} packed_index_type;
    float val_marker = std::numeric_limits<float>::infinity();
    unsigned int idx_marker = std::numeric_limits<unsigned int>::max();

    SpMVDataFormatter<float, num_PEs_per_hbm_channel, packed_data_type, packed_index_type> formatter(M);
    formatter.format_pad_marker_end_of_row(out_buffer_len,
                                           vector_buffer_len,
                                           num_hbm_channels,
                                           val_marker,
                                           idx_marker);

    std::vector<packed_data_type> reference_data_row_partition_1 =
        {{1,5}, {2,6}, {3,val_marker}, {4,0}, {val_marker,0}};
    std::vector<packed_index_type> reference_indices_row_partition_1 =
        {{0,0}, {1,2}, {2,idx_marker}, {3,0}, {idx_marker,0}};
    std::vector<packed_data_type> reference_data_row_partition_2 =
        {{7,8}, {val_marker,val_marker}};
    std::vector<packed_index_type> reference_indices_row_partition_2 =
        {{1,3}, {idx_marker,idx_marker}};

    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        formatter.get_packed_data(0, 0, 0), reference_data_row_partition_1);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        formatter.get_packed_indices(0, 0, 0), reference_indices_row_partition_1);
    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        formatter.get_packed_data(1, 0, 0), reference_data_row_partition_2);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        formatter.get_packed_indices(1, 0, 0), reference_indices_row_partition_2);

    std::cout << "test_spmv_data_formatter_row_partitioning passed" << std::endl;
}


int main(int argc, char *argv[]) {
    // Test data loader
    test_create_csr_matrix();
    test_load_csr_matrix_from_float_npz();
    test_csr_matrix_convert_from_float();

    // Test data formatter
    test_util_round_csr_matrix_dim();
    test_util_convert_csr_to_dds();
    test_util_reorder_rows_ascending_nnz();
    test_util_pack_rows();
    test_spmv_data_formatter_col_partitioning();
    test_spmv_data_formatter_row_partitioning();
}
