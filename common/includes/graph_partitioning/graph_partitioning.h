#ifndef __GRAPH_PARTITIONING_H
#define __GRAPH_PARTITIONING_H

#include <cstdint>
#include <vector>


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

#endif
