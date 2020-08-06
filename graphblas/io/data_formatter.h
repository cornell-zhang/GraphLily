#ifndef __GRAPHBLAS_IO_DATA_FORMATTER_H
#define __GRAPHBLAS_IO_DATA_FORMATTER_H

#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>

#include "./data_loader.h"


namespace graphblas {
namespace io {

/*!
 * \brief Doing src vertex partitioning (column dimension) by converting csr to dds (dense-dense-sparse).
 *
 * \param num_rows The number of rows of the original sparse matrix.
 * \param num_cols The number of columns of the original sparse matrix.
 * \param adj_data The non-zero data (CSR).
 * \param adj_indices The column indices (CSR).
 * \param adj_indptr The index pointers (CSR).
 * \param num_cols_per_partition The number of columns per partition, determined by the vector buffer length.
 *
 * \param partitioned_adj_data The partitioned non-zero data.
 * \param partitioned_adj_indices The partitioned column indices.
 * \param partitioned_adj_indptr The partitioned index pointers.
 */
template<typename data_type>
void util_convert_csr_to_dds(uint32_t num_rows,
                             uint32_t num_cols,
                             const data_type *adj_data,
                             const uint32_t *adj_indices,
                             const uint32_t *adj_indptr,
                             uint32_t num_cols_per_partition,
                             std::vector<data_type> partitioned_adj_data[],
                             std::vector<uint32_t> partitioned_adj_indices[],
                             std::vector<uint32_t> partitioned_adj_indptr[])
{
    uint32_t num_col_partitions = (num_cols + num_cols_per_partition - 1) / num_cols_per_partition;

    for (uint32_t partition_idx = 0; partition_idx < num_col_partitions; partition_idx++) {
        partitioned_adj_indptr[partition_idx].resize(num_rows + 1);
        partitioned_adj_indptr[partition_idx][0] = 0; // The first element in indptr is 0
    }

    // Perform partitioning in two passes:
    //   In the first pass, count the nnz of each partition, and resize the vectors accordingly.
    //   In the second pass, write values to the vectors.

    // The first pass
    int nnz_count[num_col_partitions];
    for (uint32_t partition_idx = 0; partition_idx < num_col_partitions; partition_idx++) {
        nnz_count[partition_idx] = 0;
    }
    for (uint32_t i = 0; i < num_rows; i++) {
        for (uint32_t j = adj_indptr[i]; j < adj_indptr[i + 1]; j++) {
            uint32_t col_idx = adj_indices[j];
            uint32_t partition_idx = col_idx / num_cols_per_partition;
            nnz_count[partition_idx]++;
        }
        for (uint32_t partition_idx = 0; partition_idx < num_col_partitions; partition_idx++) {
            partitioned_adj_indptr[partition_idx][i+1] = nnz_count[partition_idx];
        }
    }
    for (uint32_t partition_idx = 0; partition_idx < num_col_partitions; partition_idx++) {
        partitioned_adj_data[partition_idx].resize(partitioned_adj_indptr[partition_idx].back());
        partitioned_adj_indices[partition_idx].resize(partitioned_adj_indptr[partition_idx].back());
    }

    // The second pass
    int pos[num_col_partitions];
    for (uint32_t partition_idx = 0; partition_idx < num_col_partitions; partition_idx++) {
        pos[partition_idx] = 0;
    }
    for (uint32_t i = 0; i < num_rows; i++) {
        for (uint32_t j = adj_indptr[i]; j < adj_indptr[i + 1]; j++) {
            data_type val = adj_data[j];
            uint32_t col_idx = adj_indices[j];
            uint32_t partition_idx = col_idx / num_cols_per_partition;
            partitioned_adj_data[partition_idx][pos[partition_idx]] = val;
            partitioned_adj_indices[partition_idx][pos[partition_idx]] =
                col_idx - partition_idx*num_cols_per_partition;
            pos[partition_idx]++;
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


/*!
 * \brief Pack the rows.
 *
 * \param adj_data The non-zero data (CSR).
 * \param adj_indices The column indices (CSR).
 * \param adj_indptr The index pointers (CSR).
 * \param num_PEs_per_hbm_channel The number of rows to pack together.
 *
 * \param packed_adj_data The packed non-zero data.
 * \param packed_adj_indices The packed column indices.
 * \param packed_adj_indptr The packed index pointers.
 */
template<typename data_type, typename packed_data_type, typename packed_index_type>
void util_pack_rows(std::vector<data_type> const &adj_data,
                    std::vector<uint32_t> const &adj_indices,
                    std::vector<uint32_t> const &adj_indptr,
                    uint32_t num_hbm_channels,
                    uint32_t num_PEs_per_hbm_channel,
                    std::vector<packed_data_type> packed_adj_data[],
                    std::vector<packed_index_type> packed_adj_indices[],
                    std::vector<packed_index_type> packed_adj_indptr[])
{
    uint32_t num_rows = adj_indptr.size() - 1;
    uint32_t num_packs = (num_rows + num_hbm_channels*num_PEs_per_hbm_channel - 1)
                         / (num_hbm_channels*num_PEs_per_hbm_channel);
    std::vector<uint32_t> nnz_each_row(num_rows);
    for (size_t i = 0; i < num_rows; i++) {
        nnz_each_row[i] = adj_indptr[i + 1] - adj_indptr[i];
    }

    // Handle indptr
    packed_index_type tmp_indptr[num_hbm_channels];
    for (size_t c = 0; c < num_hbm_channels; c++) {
        for (size_t j = 0; j < num_PEs_per_hbm_channel; j++) {tmp_indptr[c].data[j] = 0;}
        packed_adj_indptr[c].push_back(tmp_indptr[c]);
    }
    for (size_t i = 0; i < num_packs; i++) {
        for (size_t c = 0; c < num_hbm_channels; c++) {
            for (size_t j = 0; j < num_PEs_per_hbm_channel; j++) {
                size_t row_idx = i*num_hbm_channels*num_PEs_per_hbm_channel + c*num_PEs_per_hbm_channel + j;
                if (row_idx < num_rows) {
                    tmp_indptr[c].data[j] = tmp_indptr[c].data[j] + nnz_each_row[row_idx];
                }
            }
            packed_adj_indptr[c].push_back(tmp_indptr[c]);
        }
    }

    // Handle data and indices
    for (size_t c = 0; c < num_hbm_channels; c++) {
        uint32_t max_nnz = 0;
        for (uint32_t i = 0; i < num_PEs_per_hbm_channel; i++) {
            if (max_nnz < packed_adj_indptr[c].back().data[i]) {
                max_nnz = packed_adj_indptr[c].back().data[i];
            }
        }
        packed_adj_data[c].resize(max_nnz);
        packed_adj_indices[c].resize(max_nnz);
        for (size_t j = 0; j < num_PEs_per_hbm_channel; j++) {
            uint32_t nnz_count = 0;
            for (size_t i = 0; i < num_packs; i++) {
                size_t row_idx = i*num_hbm_channels*num_PEs_per_hbm_channel + c*num_PEs_per_hbm_channel + j;
                if (row_idx >= num_rows) {
                    continue;
                }
                for (uint32_t k = adj_indptr[row_idx]; k < adj_indptr[row_idx + 1]; k++) {
                    data_type val = adj_data[k];
                    uint32_t col_idx = adj_indices[k];
                    packed_adj_data[c][nnz_count].data[j] = val;
                    packed_adj_indices[c][nnz_count].data[j] = col_idx;
                    nnz_count++;
                }
            }
        }
    }
}


template<typename data_type>
void util_pad_marker_end_of_row(std::vector<data_type> &adj_data,
                                std::vector<uint32_t> &adj_indices,
                                std::vector<uint32_t> &adj_indptr,
                                data_type val_marker,
                                uint32_t idx_marker) {
    uint32_t num_rows = adj_indptr.size() - 1;
    for (size_t row_idx = 0; row_idx < num_rows; row_idx++) {
        adj_data.insert(adj_data.begin() + adj_indptr[row_idx + 1] + row_idx, val_marker);
        adj_indices.insert(adj_indices.begin() + adj_indptr[row_idx + 1] + row_idx, idx_marker);
    }
    for (size_t row_idx = 0; row_idx < num_rows; row_idx++) {
        adj_indptr[row_idx + 1] += (row_idx + 1);
    }
}


template<typename data_type>
void util_round_csr_matrix_dim(CSRMatrix<data_type> &csr_matrix,
                               uint32_t row_divisor,
                               uint32_t col_divisor) {
    if (csr_matrix.num_rows % row_divisor != 0) {
        uint32_t num_rows_to_pad = row_divisor - csr_matrix.num_rows % row_divisor;
        for (size_t i = 0; i < num_rows_to_pad; i++) {
            csr_matrix.adj_indptr.push_back(csr_matrix.adj_indptr[csr_matrix.num_rows]);
        }
        csr_matrix.num_rows += num_rows_to_pad;
    }
    if (csr_matrix.num_cols % col_divisor != 0) {
        uint32_t num_cols_to_pad = col_divisor - csr_matrix.num_cols % col_divisor;
        csr_matrix.num_cols += num_cols_to_pad;
    }
}


/*!
 * \brief Formatter for the sparse matrix used in SpMV. It does column partitioning and row packing.
 *
 * \tparam data_type The data type of non-zero values of the sparse matrix.
 * \tparam num_PEs_per_hbm_channel The number of PEs per HBM channel.
 * \tparam packed_data_type The data type of non-zero values of the packed sparse matrix.
 * \tparam packed_index_type The data type of indices of the packed sparse matrix.
 */
template<typename data_type, uint32_t num_PEs_per_hbm_channel, typename packed_data_type, typename packed_index_type>
class SpMVDataFormatter {
private:
    /*! \brief The sparse matrix */
    CSRMatrix<data_type> csr_matrix_;
    /*! \brief The number of partitions along the row dimension */
    uint32_t num_row_partitions_;
    /*! \brief The number of partitions along the column dimension */
    uint32_t num_col_partitions_;
    /*! \brief The number of HBM channels */
    uint32_t num_hbm_channels_;

    std::vector<std::vector<packed_data_type> > formatted_adj_data;
    std::vector<std::vector<packed_index_type> > formatted_adj_indices;
    std::vector<std::vector<packed_index_type> > formatted_adj_indptr;

private:
    void _format(uint32_t out_buffer_len,
                 uint32_t vector_buffer_len,
                 uint32_t num_hbm_channels,
                 bool pad_marker_end_of_row,
                 data_type val_marker,
                 uint32_t idx_marker);

public:
    SpMVDataFormatter(CSRMatrix<data_type> const &csr_matrix) {
        this->csr_matrix_ = csr_matrix;
    }

    /*!
     * \brief Format the sparse matrix by performing column partitioning and row packing.
     * \param out_buffer_len The output buffer length, which determines the number of row partitions.
     * \param vector_buffer_len The vector buffer length, which determines the number of column partitions.
     * \param num_hbm_channels The number of HBM channels.
     */
    void format(uint32_t out_buffer_len,
                uint32_t vector_buffer_len,
                uint32_t num_hbm_channels) {
        this->_format(out_buffer_len, vector_buffer_len, num_hbm_channels, false, 0, 0);
    }

    /*!
     * \brief Compared with the format method, format_pad_marker_end_of_row gets rid of adj_indptr
     *        by padding a marker to adj_data and adj_indices to denote the end of a row.
     * \param out_buffer_len The output buffer length, which determines the number of row partitions.
     * \param vector_buffer_len The vector buffer length, which determines the number of column partitions.
     * \param num_hbm_channels The number of HBM channels.
     * \param val_marker The marker to be padded into adj_data to denote the end of a row.
     * \param idx_marker The marker to be padded into adj_indices to denote the end of a row.
     */
    void format_pad_marker_end_of_row(uint32_t out_buffer_len,
                                      uint32_t vector_buffer_len,
                                      uint32_t num_hbm_channels,
                                      data_type val_marker,
                                      uint32_t idx_marker) {
        this->_format(out_buffer_len, vector_buffer_len, num_hbm_channels, true, val_marker, idx_marker);
    }

    /*!
     * \brief Get the non-zero data of a specific partition for a specific HBM channel.
     * \param row_partition_idx The row partition index.
     * \param col_partition_idx The column partition index.
     * \param hbm_channel_idx The HBM channel index.
     * \return The non-zero data.
     */
    std::vector<packed_data_type> get_packed_data(uint32_t row_partition_idx,
                                                  uint32_t col_partition_idx,
                                                  uint32_t hbm_channel_idx) {
        return this->formatted_adj_data[row_partition_idx*this->num_col_partitions_*this->num_hbm_channels_
                                        + col_partition_idx*this->num_hbm_channels_ + hbm_channel_idx];
    }

    /*!
     * \brief Get the column indices of a specific partition for a specific HBM channel.
     * \param row_partition_idx The row partition index.
     * \param col_partition_idx The column partition index.
     * \param hbm_channel_idx The HBM channel index.
     * \return The column indices.
     */
    std::vector<packed_index_type> get_packed_indices(uint32_t row_partition_idx,
                                                      uint32_t col_partition_idx,
                                                      uint32_t hbm_channel_idx) {
        return this->formatted_adj_indices[row_partition_idx*this->num_col_partitions_*this->num_hbm_channels_
                                           + col_partition_idx*this->num_hbm_channels_ + hbm_channel_idx];
    }

    /*!
     * \brief Get the index pointers of a specific partition for a specific HBM channel.
     * \param row_partition_idx The row partition index.
     * \param col_partition_idx The column partition index.
     * \param hbm_channel_idx The HBM channel index.
     * \return The index pointers.
     */
    std::vector<packed_index_type> get_packed_indptr(uint32_t row_partition_idx,
                                                     uint32_t col_partition_idx,
                                                     uint32_t hbm_channel_idx) {
        return this->formatted_adj_indptr[row_partition_idx*this->num_col_partitions_*this->num_hbm_channels_
                                          + col_partition_idx*this->num_hbm_channels_ + hbm_channel_idx];
    }
};


template<typename data_type, uint32_t num_PEs_per_hbm_channel, typename packed_data_type, typename packed_index_type>
void SpMVDataFormatter<data_type, num_PEs_per_hbm_channel, packed_data_type, packed_index_type>::
_format(uint32_t out_buffer_len,
        uint32_t vector_buffer_len,
        uint32_t num_hbm_channels,
        bool pad_marker_end_of_row,
        data_type val_marker,
        uint32_t idx_marker)
{
    if (this->csr_matrix_.num_rows % (num_PEs_per_hbm_channel * num_hbm_channels) != 0) {
        std::cout << "The number of rows of the sparse matrix should divide "
                    << num_PEs_per_hbm_channel * num_hbm_channels << ". "
                    << "Please use graphblas::io::util_round_csr_matrix_dim. "
                    << "Exit!" <<std::endl;
        exit(EXIT_FAILURE);
    }
    if (this->csr_matrix_.num_cols % num_PEs_per_hbm_channel != 0) {
        std::cout << "The number of columns of the sparse matrix should divide "
                    << num_PEs_per_hbm_channel << ". "
                    << "Please use graphblas::io::util_round_csr_matrix_dim. "
                    << "Exit!" <<std::endl;
        exit(EXIT_FAILURE);
    }
    assert((out_buffer_len % (num_PEs_per_hbm_channel * num_hbm_channels) == 0));
    assert((vector_buffer_len % num_PEs_per_hbm_channel == 0));
    this->num_row_partitions_ = (this->csr_matrix_.num_rows + out_buffer_len - 1) / out_buffer_len;
    this->num_col_partitions_ = (this->csr_matrix_.num_cols + vector_buffer_len - 1) / vector_buffer_len;
    this->formatted_adj_data.resize(this->num_row_partitions_*this->num_col_partitions_*num_hbm_channels);
    this->formatted_adj_indices.resize(this->num_row_partitions_*this->num_col_partitions_*num_hbm_channels);
    this->formatted_adj_indptr.resize(this->num_row_partitions_*this->num_col_partitions_*num_hbm_channels);
    for (size_t j = 0; j < this->num_row_partitions_; j++) {
        std::vector<data_type> partitioned_adj_data[this->num_col_partitions_];
        std::vector<uint32_t> partitioned_adj_indices[this->num_col_partitions_];
        std::vector<uint32_t> partitioned_adj_indptr[this->num_col_partitions_];
        uint32_t num_rows = out_buffer_len;
        if (j == (this->num_row_partitions_ - 1)) {
            num_rows = this->csr_matrix_.num_rows - (this->num_row_partitions_ - 1) * out_buffer_len;
        }
        std::vector<uint32_t> adj_indptr_slice(this->csr_matrix_.adj_indptr.begin() + j*out_buffer_len,
            this->csr_matrix_.adj_indptr.begin() + j*out_buffer_len + num_rows + 1);
        uint32_t offset = this->csr_matrix_.adj_indptr[j * out_buffer_len];
        for (auto &x : adj_indptr_slice) x -= offset;
        util_convert_csr_to_dds<data_type>(num_rows,
                                           this->csr_matrix_.num_cols,
                                           this->csr_matrix_.adj_data.data() + offset,
                                           this->csr_matrix_.adj_indices.data() + offset,
                                           adj_indptr_slice.data(),
                                           vector_buffer_len,
                                           partitioned_adj_data,
                                           partitioned_adj_indices,
                                           partitioned_adj_indptr);
        for (size_t i = 0; i < this->num_col_partitions_; i++) {
            if (pad_marker_end_of_row) {
                util_pad_marker_end_of_row<data_type>(partitioned_adj_data[i],
                                                      partitioned_adj_indices[i],
                                                      partitioned_adj_indptr[i],
                                                      val_marker,
                                                      idx_marker);
            }
            util_pack_rows<data_type, packed_data_type, packed_index_type>(
                partitioned_adj_data[i],
                partitioned_adj_indices[i],
                partitioned_adj_indptr[i],
                num_hbm_channels,
                num_PEs_per_hbm_channel,
                &(this->formatted_adj_data[j*this->num_col_partitions_*num_hbm_channels + i*num_hbm_channels]),
                &(this->formatted_adj_indices[j*this->num_col_partitions_*num_hbm_channels + i*num_hbm_channels]),
                &(this->formatted_adj_indptr[j*this->num_col_partitions_*num_hbm_channels + i*num_hbm_channels])
            );
        }
    }
    this->num_hbm_channels_ = num_hbm_channels;
}


} // namespace io
} // namespace graphblas

#endif // __GRAPHBLAS_IO_DATA_FORMATTER_H
