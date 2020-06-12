#include <cstdlib>
#include <iostream>
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

    uint32_t num_col_partitions = 2;
    std::vector<float> partitioned_adj_data[num_col_partitions];
    std::vector<uint32_t> partitioned_adj_indices[num_col_partitions];
    std::vector<uint32_t> partitioned_adj_indptr[num_col_partitions];

    util_convert_csr_to_dds<float>(num_rows,
                                   num_cols,
                                   adj_data,
                                   adj_indices,
                                   adj_indptr,
                                   num_col_partitions,
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


int main(int argc, char *argv[]) {
    test_util_convert_csr_to_dds();
}
