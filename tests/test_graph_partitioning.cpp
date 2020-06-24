#include <cstdlib>
#include <iostream>
#include <limits>
#include "graph_partitioning.h"


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


void test_util_convert_csr_to_dds() {
    uint32_t num_rows = 5;
    uint32_t num_cols = 5;
    uint32_t nnz = 9;
    float adj_data[nnz] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint32_t adj_indices[nnz] = {0, 1, 2, 3, 0, 2, 1, 3, 2};
    uint32_t adj_indptr[num_rows + 1] = {0, 4, 6, 7, 8, 9};

    // The sparse matrix is:
    //     [[1, 2, 3, 4, 0],
    //      [5, 0, 6, 0, 0],
    //      [0, 7, 0, 0, 0],
    //      [0, 0, 0, 8, 0],
    //      [0, 0, 9, 0, 0]]

    uint32_t num_cols_per_partition = 3;
    uint32_t num_col_partitions = (num_cols + num_cols_per_partition - 1) / num_cols_per_partition;
    std::vector<float> partitioned_adj_data[num_col_partitions];
    std::vector<uint32_t> partitioned_adj_indices[num_col_partitions];
    std::vector<uint32_t> partitioned_adj_indptr[num_col_partitions];

    util_convert_csr_to_dds<float>(num_rows,
                                   num_cols,
                                   adj_data,
                                   adj_indices,
                                   adj_indptr,
                                   num_cols_per_partition,
                                   partitioned_adj_data,
                                   partitioned_adj_indices,
                                   partitioned_adj_indptr);

    std::vector<float> reference_partition_1_adj_data = {1, 2, 3, 5, 6, 7, 9};
    std::vector<float> reference_partition_2_adj_data = {4, 8};
    std::vector<uint32_t> reference_partition_1_adj_indices = {0, 1, 2, 0, 2, 1, 2};
    std::vector<uint32_t> reference_partition_2_adj_indices = {0, 0};
    std::vector<uint32_t> reference_partition_1_adj_indptr = {0, 3, 5, 6, 6, 7};
    std::vector<uint32_t> reference_partition_2_adj_indptr = {0, 1, 1, 1, 2, 2};

    check_vector_equal<float>(partitioned_adj_data[0], reference_partition_1_adj_data);
    check_vector_equal<float>(partitioned_adj_data[1], reference_partition_2_adj_data);
    check_vector_equal<uint32_t>(partitioned_adj_indices[0], reference_partition_1_adj_indices);
    check_vector_equal<uint32_t>(partitioned_adj_indices[1], reference_partition_2_adj_indices);
    check_vector_equal<uint32_t>(partitioned_adj_indptr[0], reference_partition_1_adj_indptr);
    check_vector_equal<uint32_t>(partitioned_adj_indptr[1], reference_partition_2_adj_indptr);

    std::cout << "Test passed" << std::endl;
}


void test_util_reorder_rows_ascending_nnz() {
    std::vector<float> adj_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<uint32_t> adj_indices = {0, 1, 2, 3, 0, 2, 1, 3, 2};
    std::vector<uint32_t> adj_indptr = {0, 4, 6, 7, 8, 9};

    // The sparse matrix is:
    //     [[1, 2, 3, 4, 0],
    //      [5, 0, 6, 0, 0],
    //      [0, 7, 0, 0, 0],
    //      [0, 0, 0, 8, 0],
    //      [0, 0, 9, 0, 0]]

    std::vector<float> reordered_adj_data;
    std::vector<uint32_t> reordered_adj_indices;
    std::vector<uint32_t> reordered_adj_indptr;

    util_reorder_rows_ascending_nnz<float>(adj_data,
                                           adj_indices,
                                           adj_indptr,
                                           reordered_adj_data,
                                           reordered_adj_indices,
                                           reordered_adj_indptr);

    // After reordering, the sparse matrix is:
    //     [[0, 7, 0, 0, 0],
    //      [0, 0, 0, 8, 0],
    //      [0, 0, 9, 0, 0],
    //      [5, 0, 6, 0, 0],
    //      [1, 2, 3, 4, 0]]

    std::vector<float> reference_reordered_adj_data = {7, 8, 9, 5, 6, 1, 2, 3, 4};
    std::vector<uint32_t> reference_reordered_adj_indices = {1, 3, 2, 0, 2, 0, 1, 2, 3};
    std::vector<uint32_t> reference_reordered_adj_indptr = {0, 1, 2, 3, 5, 9};

    check_vector_equal<float>(reordered_adj_data, reference_reordered_adj_data);
    check_vector_equal<uint32_t>(reordered_adj_indices, reference_reordered_adj_indices);
    check_vector_equal<uint32_t>(reordered_adj_indptr, reference_reordered_adj_indptr);

    std::cout << "Test passed" << std::endl;
}


void test_util_pack_rows() {
    std::vector<float> adj_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<uint32_t> adj_indices = {0, 1, 2, 3, 0, 2, 1, 3, 2, 1, 3, 0};
    std::vector<uint32_t> adj_indptr = {0, 4, 6, 7, 8, 9, 11, 12};

    // The sparse matrix is:
    //     [[1, 2, 3, 4, 0],
    //      [5, 0, 6, 0, 0],
    //      [0, 7, 0, 0, 0],
    //      [0, 0, 0, 8, 0],
    //      [0, 0, 9, 0, 0],
    //      [0, 10,0, 11,0],
    //      [12,0, 0, 0, 0]]

    const uint32_t num_hbm_channels = 2;
    const uint32_t num_PEs_per_hbm_channel = 2;

    typedef struct packed_data_type_ {
        float data[num_PEs_per_hbm_channel];
    } packed_data_type;

    typedef struct packed_index_type_ {
        uint32_t data[num_PEs_per_hbm_channel];
    } packed_index_type;

    std::vector<packed_data_type> packed_adj_data[num_hbm_channels];
    std::vector<packed_index_type> packed_adj_indices[num_hbm_channels];
    std::vector<packed_index_type> packed_adj_indptr[num_hbm_channels];

    util_pack_rows<float, packed_data_type, packed_index_type>(adj_data,
                                                               adj_indices,
                                                               adj_indptr,
                                                               num_hbm_channels,
                                                               num_PEs_per_hbm_channel,
                                                               packed_adj_data,
                                                               packed_adj_indices,
                                                               packed_adj_indptr);

    std::vector<packed_data_type> reference_packed_adj_data_channel_1 = {{1,5}, {2,6}, {3,10}, {4,11}, {9,0}};
    std::vector<packed_index_type> reference_packed_adj_indices_channel_1 = {{0,0}, {1,2}, {2,1}, {3,3}, {2,0}};
    std::vector<packed_index_type> reference_packed_adj_indptr_channel_1 = {{0,0}, {4,2}, {5,4}};
    std::vector<packed_data_type> reference_packed_adj_data_channel_2 = {{7,8}, {12,0}};
    std::vector<packed_index_type> reference_packed_adj_indices_channel_2 = {{1,3}, {0,0}};
    std::vector<packed_index_type> reference_packed_adj_indptr_channel_2 = {{0,0}, {1,1}, {2,1}};

    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        packed_adj_data[0], reference_packed_adj_data_channel_1);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        packed_adj_indices[0], reference_packed_adj_indices_channel_1);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        packed_adj_indptr[0], reference_packed_adj_indptr_channel_1);
    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        packed_adj_data[1], reference_packed_adj_data_channel_2);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        packed_adj_indices[1], reference_packed_adj_indices_channel_2);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        packed_adj_indptr[1], reference_packed_adj_indptr_channel_2);

    std::cout << "Test passed" << std::endl;
}


void test_spmv_data_formatter_no_pad_marker() {
    uint32_t num_rows = 7;
    uint32_t num_cols = 10;
    std::vector<float> adj_data = {1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 5, 6, 7, 7, 8, 8, 9, 9, 10, 11, 10, 11, 12, 12};
    std::vector<uint32_t> adj_indices = {0, 1, 2, 3, 5, 6, 7, 8, 0, 2, 5, 7, 1, 6, 3, 8, 2, 7, 1, 3, 6, 8, 0, 5};
    std::vector<uint32_t> adj_indptr = {0, 8, 12, 14, 16, 18, 22, 24};

    // The sparse matrix is:
    //     [[1, 2, 3, 4, 0, 1, 2, 3, 4, 0],
    //      [5, 0, 6, 0, 0, 5, 0, 6, 0, 0],
    //      [0, 7, 0, 0, 0, 0, 7, 0, 0, 0],
    //      [0, 0, 0, 8, 0, 0, 0, 0, 8, 0],
    //      [0, 0, 9, 0, 0, 0, 0, 9, 0, 0],
    //      [0, 10,0, 11,0, 0, 10,0, 11,0],
    //      [12,0, 0, 0, 0, 12,0, 0, 0, 0]]

    const uint32_t vector_buffer_len = 5;
    const uint32_t num_hbm_channels = 2;
    const uint32_t num_PEs_per_hbm_channel = 2;

    typedef struct packed_data_type_ {
        float data[num_PEs_per_hbm_channel];
    } packed_data_type;

    typedef struct packed_index_type_ {
        uint32_t data[num_PEs_per_hbm_channel];
    } packed_index_type;

    SpMVDataFormatter<float, num_PEs_per_hbm_channel, packed_data_type, packed_index_type>
        formatter(num_rows, num_cols, adj_data.data(), adj_indices.data(), adj_indptr.data());
    formatter.format(vector_buffer_len, num_hbm_channels);

    // The two partitions are the same
    std::vector<packed_data_type> reference_packed_adj_data_channel_1 = {{1,5}, {2,6}, {3,10}, {4,11}, {9,0}};
    std::vector<packed_index_type> reference_packed_adj_indices_channel_1 = {{0,0}, {1,2}, {2,1}, {3,3}, {2,0}};
    std::vector<packed_index_type> reference_packed_adj_indptr_channel_1 = {{0,0}, {4,2}, {5,4}};
    std::vector<packed_data_type> reference_packed_adj_data_channel_2 = {{7,8}, {12,0}};
    std::vector<packed_index_type> reference_packed_adj_indices_channel_2 = {{1,3}, {0,0}};
    std::vector<packed_index_type> reference_packed_adj_indptr_channel_2 = {{0,0}, {1,1}, {2,1}};

    auto partition_1_channel_1_data = formatter.get_packed_data(0, 0);
    auto partition_1_channel_1_indices = formatter.get_packed_indices(0, 0);
    auto partition_1_channel_1_indptr = formatter.get_packed_indptr(0, 0);
    auto partition_1_channel_2_data = formatter.get_packed_data(0, 1);
    auto partition_1_channel_2_indices = formatter.get_packed_indices(0, 1);
    auto partition_1_channel_2_indptr = formatter.get_packed_indptr(0, 1);
    auto partition_2_channel_1_data = formatter.get_packed_data(1, 0);
    auto partition_2_channel_1_indices = formatter.get_packed_indices(1, 0);
    auto partition_2_channel_1_indptr = formatter.get_packed_indptr(1, 0);
    auto partition_2_channel_2_data = formatter.get_packed_data(1, 1);
    auto partition_2_channel_2_indices = formatter.get_packed_indices(1, 1);
    auto partition_2_channel_2_indptr = formatter.get_packed_indptr(1, 1);

    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        partition_1_channel_1_data, reference_packed_adj_data_channel_1);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        partition_1_channel_1_indices, reference_packed_adj_indices_channel_1);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        partition_1_channel_1_indptr, reference_packed_adj_indptr_channel_1);
    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        partition_1_channel_2_data, reference_packed_adj_data_channel_2);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        partition_1_channel_2_indices, reference_packed_adj_indices_channel_2);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        partition_1_channel_2_indptr, reference_packed_adj_indptr_channel_2);
    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        partition_2_channel_1_data, reference_packed_adj_data_channel_1);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        partition_2_channel_1_indices, reference_packed_adj_indices_channel_1);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        partition_2_channel_1_indptr, reference_packed_adj_indptr_channel_1);
    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        partition_2_channel_2_data, reference_packed_adj_data_channel_2);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        partition_2_channel_2_indices, reference_packed_adj_indices_channel_2);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        partition_2_channel_2_indptr, reference_packed_adj_indptr_channel_2);

    std::cout << "Test passed" << std::endl;
}


void test_spmv_data_formatter_pad_marker() {
    uint32_t num_rows = 7;
    uint32_t num_cols = 5;
    std::vector<float> adj_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<uint32_t> adj_indices = {0, 1, 2, 3, 0, 2, 1, 3, 2, 1, 3, 0};
    std::vector<uint32_t> adj_indptr = {0, 4, 6, 7, 8, 9, 11, 12};

    // The sparse matrix is:
    //     [[1, 2, 3, 4, 0],
    //      [5, 0, 6, 0, 0],
    //      [0, 7, 0, 0, 0],
    //      [0, 0, 0, 8, 0],
    //      [0, 0, 9, 0, 0],
    //      [0, 10,0, 11,0],
    //      [12,0, 0, 0, 0]]

    const uint32_t vector_buffer_len = 5;
    const uint32_t num_hbm_channels = 2;
    const uint32_t num_PEs_per_hbm_channel = 2;

    typedef struct packed_data_type_ {
        float data[num_PEs_per_hbm_channel];
    } packed_data_type;

    typedef struct packed_index_type_ {
        uint32_t data[num_PEs_per_hbm_channel];
    } packed_index_type;

    float val_marker = std::numeric_limits<float>::infinity();
    unsigned int idx_marker = std::numeric_limits<unsigned int>::max();

    SpMVDataFormatter<float, num_PEs_per_hbm_channel, packed_data_type, packed_index_type>
        formatter(num_rows, num_cols, adj_data.data(), adj_indices.data(), adj_indptr.data());
    formatter.format_pad_marker_end_of_row(vector_buffer_len, num_hbm_channels, val_marker, idx_marker);

    std::vector<packed_data_type> reference_packed_adj_data_channel_1 =
        {{1,5}, {2,6}, {3,val_marker}, {4,10}, {val_marker,11}, {9,val_marker}, {val_marker,0}};
    std::vector<packed_index_type> reference_packed_adj_indices_channel_1 =
        {{0,0}, {1,2}, {2,idx_marker}, {3,1}, {idx_marker,3}, {2,idx_marker}, {idx_marker,0}};
    std::vector<packed_data_type> reference_packed_adj_data_channel_2 =
        {{7,8}, {val_marker,val_marker}, {12,0}, {val_marker,0}};
    std::vector<packed_index_type> reference_packed_adj_indices_channel_2 =
        {{1,3}, {idx_marker,idx_marker}, {0,0}, {idx_marker,0}};

    auto partition_1_channel_1_data = formatter.get_packed_data(0, 0);
    auto partition_1_channel_1_indices = formatter.get_packed_indices(0, 0);
    auto partition_1_channel_2_data = formatter.get_packed_data(0, 1);
    auto partition_1_channel_2_indices = formatter.get_packed_indices(0, 1);

    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        partition_1_channel_1_data, reference_packed_adj_data_channel_1);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        partition_1_channel_1_indices, reference_packed_adj_indices_channel_1);
    check_packed_vector_equal<packed_data_type, num_PEs_per_hbm_channel>(
        partition_1_channel_2_data, reference_packed_adj_data_channel_2);
    check_packed_vector_equal<packed_index_type, num_PEs_per_hbm_channel>(
        partition_1_channel_2_indices, reference_packed_adj_indices_channel_2);

    std::cout << "Test passed" << std::endl;
}


int main(int argc, char *argv[]) {
    test_util_convert_csr_to_dds();
    test_util_reorder_rows_ascending_nnz();
    test_util_pack_rows();
    test_spmv_data_formatter_no_pad_marker();
    test_spmv_data_formatter_pad_marker();
}
