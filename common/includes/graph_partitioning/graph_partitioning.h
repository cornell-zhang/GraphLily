#ifndef __GRAPH_PARTITIONING_H
#define __GRAPH_PARTITIONING_H

#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>


/*!
 * \brief Doing src vertex partitioning (column dimension) by converting csr to dds (dense-dense-sparse).
 *
 * \param num_rows The number of rows of the original sparse matrix.
 * \param num_cols The number of columns of the original sparse matrix.
 * \param adj_data The non-zero data (CSR).
 * \param adj_indices The column indices (CSR).
 * \param adj_indptr The index pointers (CSR).
 * \param num_col_partitions Number of partitions along the column dimension.
 *
 * \param partitioned_adj_data The partitioned non-zero data.
 * \param partitioned_adj_indices The partitioned column indices.
 * \param partitioned_adj_indptr The partitioned index pointers.
 */
template<typename data_type>
void util_convert_csr_to_dds(const uint32_t num_rows,
                             const uint32_t num_cols,
                             const data_type *adj_data,
                             const uint32_t *adj_indices,
                             const uint32_t *adj_indptr,
                             const uint32_t num_col_partitions,
                             std::vector<data_type> partitioned_adj_data[],
                             std::vector<uint32_t> partitioned_adj_indices[],
                             std::vector<uint32_t> partitioned_adj_indptr[])
{
    uint32_t num_cols_per_partition = (num_cols + num_col_partitions - 1) / num_col_partitions;

    // The first element in indptr is 0
    for (uint32_t partition_idx = 0; partition_idx < num_col_partitions; partition_idx++) {
        partitioned_adj_indptr[partition_idx].push_back(0);
    }

    // Iterate the original CSR matrix row by row and perform partitioning
    for (uint32_t i = 0; i < num_rows; i++) {
        for (uint32_t j = adj_indptr[i]; j < adj_indptr[i + 1]; j++) {
            data_type val = adj_data[j];
            uint32_t col_idx = adj_indices[j];
            unsigned partition_idx = col_idx / num_cols_per_partition;
            partitioned_adj_data[partition_idx].push_back(val);
            partitioned_adj_indices[partition_idx].push_back(col_idx - partition_idx*num_cols_per_partition);
        }
        for (uint32_t partition_idx = 0; partition_idx < num_col_partitions; partition_idx++) {
            partitioned_adj_indptr[partition_idx].push_back(partitioned_adj_data[partition_idx].size());
        }
    }
}


template<typename T>
std::vector<T> argsort(std::vector<T> const &nums) {
    int n = nums.size();
    std::vector<T> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&nums](int i, int j){return nums[i] < nums[j];});
    return indices;
}


/*!
 * \brief Reorder the rows in ascending nnz.
 *
 * \param adj_data The non-zero data (CSR).
 * \param adj_indices The column indices (CSR).
 * \param adj_indptr The index pointers (CSR).
 *
 * \param reordered_adj_data The reordered non-zero data.
 * \param reordered_adj_indices The reordered column indices.
 * \param reordered_adj_indptr The reordered index pointers.
 */
template<typename data_type>
void util_reorder_rows_ascending_nnz(std::vector<data_type> const &adj_data,
                                     std::vector<uint32_t> const &adj_indices,
                                     std::vector<uint32_t> const &adj_indptr,
                                     std::vector<data_type> &reordered_adj_data,
                                     std::vector<uint32_t> &reordered_adj_indices,
                                     std::vector<uint32_t> &reordered_adj_indptr)
{
    uint32_t num_rows = adj_indptr.size() - 1;
    std::vector<uint32_t> nnz_each_row(num_rows);
    for (size_t i = 0; i < num_rows; i++) {
        nnz_each_row[i] = adj_indptr[i + 1] - adj_indptr[i];
    }

    // Get the row indices sorted by ascending nnz
    std::vector<uint32_t> rows_ascending_nnz = argsort(nnz_each_row);

    // The first element in indptr is 0
    reordered_adj_indptr.push_back(0);

    // Iterate the original CSR matrix row by row and perform reordering
    for (size_t i = 0; i < num_rows; i++) {
        uint32_t row_id = rows_ascending_nnz[i];
        for (uint32_t j = adj_indptr[row_id]; j < adj_indptr[row_id + 1]; j++) {
            data_type val = adj_data[j];
            uint32_t col_idx = adj_indices[j];
            reordered_adj_data.push_back(val);
            reordered_adj_indices.push_back(col_idx);
        }
        reordered_adj_indptr.push_back(reordered_adj_indptr.back() + nnz_each_row[row_id]);
    }
}


void util_pack_rows();


void util_prepare_spmv_adj();

#endif
