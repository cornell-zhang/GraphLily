#ifndef __GRAPHBLAS_IO_DATA_FORMATTER_H
#define __GRAPHBLAS_IO_DATA_FORMATTER_H

#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>

#include "xcl2.hpp"  // use aligned_allocator

#include "../global.h"
#include "./data_loader.h"


namespace graphblas {
namespace io {

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


template<typename data_type>
void util_normalize_csr_matrix_by_outdegree(CSRMatrix<data_type> &csr_matrix) {
    std::vector<uint32_t> nnz_each_col(csr_matrix.num_cols);
    std::fill(nnz_each_col.begin(), nnz_each_col.end(), 0);
    for (auto col_idx : csr_matrix.adj_indices) {
        nnz_each_col[col_idx]++;
    }
    for (size_t row_idx = 0; row_idx < csr_matrix.num_rows; row_idx++) {
        uint32_t start = csr_matrix.adj_indptr[row_idx];
        uint32_t end = csr_matrix.adj_indptr[row_idx + 1];
        for (size_t i = start; i < end; i++) {
            uint32_t col_idx = csr_matrix.adj_indices[i];
            csr_matrix.adj_data[i] = 1.0 / nnz_each_col[col_idx];
        }
    }
}


template<typename data_type>
void util_pad_marker_end_of_row_no_skip_empty_rows(std::vector<data_type> &adj_data,
                                                   std::vector<uint32_t> &adj_indices,
                                                   std::vector<uint32_t> &adj_indptr,
                                                   uint32_t idx_marker) {
    uint32_t num_rows = adj_indptr.size() - 1;
    std::vector<data_type> adj_data_swap(adj_data.size() + num_rows);
    std::vector<uint32_t> adj_indices_swap(adj_indices.size() + num_rows);
    data_type val_marker = 1;
    uint32_t count = 0;
    for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
        uint32_t start = adj_indptr[row_idx];
        uint32_t end = adj_indptr[row_idx + 1];
        for (uint32_t i = start; i < end; i++) {
            adj_data_swap[count] = adj_data[i];
            adj_indices_swap[count] = adj_indices[i];
            count++;
        }
        adj_data_swap[count] = val_marker;
        adj_indices_swap[count] = idx_marker;
        count++;
    }
    adj_data = adj_data_swap;
    adj_indices = adj_indices_swap;
    for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
        adj_indptr[row_idx + 1] += (row_idx + 1);
    }
}


template<typename data_type>
void util_pad_marker_end_of_row_skip_empty_rows(std::vector<data_type> &adj_data,
                                                std::vector<uint32_t> &adj_indices,
                                                std::vector<uint32_t> &adj_indptr,
                                                uint32_t idx_marker,
                                                uint32_t interleave_stride)
{
    uint32_t num_rows = adj_indptr.size() - 1;
    assert(num_rows % interleave_stride == 0);

    bool row_is_empty[num_rows];
    for (size_t row_idx = 0; row_idx < num_rows; row_idx++) {
        if (row_idx < interleave_stride) {
            // The first few rows should be inserted a marker whether empty or not
            row_is_empty[row_idx] = false;
        } else {
            uint32_t start = adj_indptr[row_idx];
            uint32_t end = adj_indptr[row_idx + 1];
            if (end == start) {
                row_is_empty[row_idx] = true;
            } else {
                row_is_empty[row_idx] = false;
            }
        }
    }

    uint32_t cumulative_sum_nonempty_rows[num_rows];
    cumulative_sum_nonempty_rows[0] = !row_is_empty[0];
    for (size_t row_idx = 1; row_idx < num_rows; row_idx++) {
        cumulative_sum_nonempty_rows[row_idx] = cumulative_sum_nonempty_rows[row_idx - 1]
                                                + !row_is_empty[row_idx];
    }

    data_type val_marker[num_rows];
    for (size_t row_idx = 0; row_idx < num_rows; row_idx++) {
        if (!row_is_empty[row_idx]) {
            val_marker[row_idx] = 1;
        } else {
            val_marker[row_idx] = 0;
        }
    }
    for (size_t k = 0; k < interleave_stride; k++) {
        for (size_t row_idx = k; row_idx < num_rows; ) {
            size_t next = row_idx + interleave_stride;
            if (!row_is_empty[row_idx]) {
                while(next < num_rows && row_is_empty[next]) {
                    val_marker[row_idx]++;
                    next += interleave_stride;
                }
            }
            row_idx = next;
        }
    }

    uint32_t total_num_nonempty_rows = cumulative_sum_nonempty_rows[num_rows - 1];
    std::vector<data_type> adj_data_swap(adj_data.size() + total_num_nonempty_rows);
    std::vector<uint32_t> adj_indices_swap(adj_indices.size() + total_num_nonempty_rows);
    uint32_t count = 0;
    for (size_t row_idx = 0; row_idx < num_rows; row_idx++) {
        if (!row_is_empty[row_idx]) {
            uint32_t start = adj_indptr[row_idx];
            uint32_t end = adj_indptr[row_idx + 1];
            for (size_t i = start; i < end; i++) {
                adj_data_swap[count] = adj_data[i];
                adj_indices_swap[count] = adj_indices[i];
                count++;
            }
            adj_data_swap[count] = val_marker[row_idx];
            adj_indices_swap[count] = idx_marker;
            count++;
        }
    }
    assert(count == adj_data_swap.size());

    adj_data = adj_data_swap;
    adj_indices = adj_indices_swap;
    for (size_t row_idx = 0; row_idx < num_rows; row_idx++) {
        adj_indptr[row_idx + 1] += cumulative_sum_nonempty_rows[row_idx];
    }
}


template<typename data_type>
void util_pad_marker_end_of_row(std::vector<data_type> &adj_data,
                                std::vector<uint32_t> &adj_indices,
                                std::vector<uint32_t> &adj_indptr,
                                uint32_t idx_marker,
                                uint32_t interleave_stride,
                                bool skip_empty_rows=false) {
    if (skip_empty_rows) {
        util_pad_marker_end_of_row_skip_empty_rows(adj_data, adj_indices, adj_indptr,
                                                   idx_marker, interleave_stride);
    } else {
        util_pad_marker_end_of_row_no_skip_empty_rows(adj_data, adj_indices, adj_indptr, idx_marker);
    }
}


//--------------------------
// Variants of CSR for SpMV
//--------------------------

// CPSR (cyclic packed streams of rows) format.
template<typename data_type, uint32_t num_PEs_per_hbm_channel>
struct CPSRMatrix {
    uint32_t num_row_partitions;
    uint32_t num_col_partitions;
    uint32_t num_hbm_channels;
    bool skip_empty_rows;

    using packed_val_t = struct {data_type data[num_PEs_per_hbm_channel];};
    using packed_idx_t = struct {uint32_t data[num_PEs_per_hbm_channel];};

    std::vector<std::vector<packed_val_t> > formatted_adj_data;
    std::vector<std::vector<packed_idx_t> > formatted_adj_indices;
    std::vector<std::vector<packed_idx_t> > formatted_adj_indptr;

    /*!
     * \brief Get the non-zero data of a specific partition for a specific HBM channel.
     */
    std::vector<packed_val_t> get_packed_data(uint32_t row_partition_idx,
                                              uint32_t col_partition_idx,
                                              uint32_t hbm_channel_idx) {
        return this->formatted_adj_data[row_partition_idx*this->num_col_partitions*this->num_hbm_channels
                                        + col_partition_idx*this->num_hbm_channels + hbm_channel_idx];
    }

    /*!
     * \brief Get the column indices of a specific partition for a specific HBM channel.
     */
    std::vector<packed_idx_t> get_packed_indices(uint32_t row_partition_idx,
                                                 uint32_t col_partition_idx,
                                                 uint32_t hbm_channel_idx) {
        return this->formatted_adj_indices[row_partition_idx*this->num_col_partitions*this->num_hbm_channels
                                           + col_partition_idx*this->num_hbm_channels + hbm_channel_idx];
    }

    /*!
     * \brief Get the indptr of a specific partition for a specific HBM channel.
     */
    std::vector<packed_idx_t> get_packed_indptr(uint32_t row_partition_idx,
                                                uint32_t col_partition_idx,
                                                uint32_t hbm_channel_idx) {
        return this->formatted_adj_indptr[row_partition_idx*this->num_col_partitions*this->num_hbm_channels
                                          + col_partition_idx*this->num_hbm_channels + hbm_channel_idx];
    }
};


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
std::vector<T> _argsort(std::vector<T> const &nums) {
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
    std::vector<uint32_t> rows_ascending_nnz = _argsort(nnz_each_row);

    // The first element in indptr is 0
    reordered_adj_indptr.push_back(0);

    // Iterate the original CSR matrix row by row and perform reordering
    for (size_t i = 0; i < num_rows; i++) {
        uint32_t row_idx = rows_ascending_nnz[i];
        for (uint32_t j = adj_indptr[row_idx]; j < adj_indptr[row_idx + 1]; j++) {
            data_type val = adj_data[j];
            uint32_t col_idx = adj_indices[j];
            reordered_adj_data.push_back(val);
            reordered_adj_indices.push_back(col_idx);
        }
        reordered_adj_indptr.push_back(reordered_adj_indptr.back() + nnz_each_row[row_idx]);
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
template<typename data_type, typename packed_val_t, typename packed_idx_t>
void util_pack_rows(std::vector<data_type> const &adj_data,
                    std::vector<uint32_t> const &adj_indices,
                    std::vector<uint32_t> const &adj_indptr,
                    uint32_t num_hbm_channels,
                    uint32_t num_PEs_per_hbm_channel,
                    std::vector<packed_val_t> packed_adj_data[],
                    std::vector<packed_idx_t> packed_adj_indices[],
                    std::vector<packed_idx_t> packed_adj_indptr[])
{
    uint32_t num_rows = adj_indptr.size() - 1;
    uint32_t num_packs = (num_rows + num_hbm_channels*num_PEs_per_hbm_channel - 1)
                         / (num_hbm_channels*num_PEs_per_hbm_channel);
    std::vector<uint32_t> nnz_each_row(num_rows);
    for (size_t i = 0; i < num_rows; i++) {
        nnz_each_row[i] = adj_indptr[i + 1] - adj_indptr[i];
    }

    // Handle indptr
    packed_idx_t tmp_indptr[num_hbm_channels];
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


/*!
 * \brief Convert a sparse matrix from CSR to CPSR.
 *
 * \tparam data_type The data type of non-zero values of the sparse matrix.
 * \tparam num_PEs_per_hbm_channel The number of PEs per HBM channel, i.e., the pack size.
 *
 * \param csr_matrix The input matrix in CSR format.
 * \param idx_marker The marker to be padded into adj_indices to denote the end of a row.
 * \param out_buf_len The output buffer length, which determines the number of row partitions.
 * \param vec_buf_len The vector buffer length, which determines the number of column partitions.
 * \param num_hbm_channels The number of HBM channels.
 * \param skip_empty_rows Whether skip empty rows or not.
 *
 * \return The output matrix in CPSR format.
 */
template <typename data_type, uint32_t num_PEs_per_hbm_channel>
CPSRMatrix<data_type, num_PEs_per_hbm_channel> csr2cpsr(CSRMatrix<data_type> const &csr_matrix,
                                                        uint32_t idx_marker,
                                                        uint32_t out_buf_len,
                                                        uint32_t vec_buf_len,
                                                        uint32_t num_hbm_channels,
                                                        bool skip_empty_rows)
{
    if (csr_matrix.num_rows % (num_PEs_per_hbm_channel * num_hbm_channels) != 0) {
        std::cout << "The number of rows of the sparse matrix should divide "
                  << num_PEs_per_hbm_channel * num_hbm_channels << ". "
                  << "Please use graphblas::io::util_round_csr_matrix_dim. "
                  << "Exit!" <<std::endl;
        exit(EXIT_FAILURE);
    }
    if (csr_matrix.num_cols % num_PEs_per_hbm_channel != 0) {
        std::cout << "The number of columns of the sparse matrix should divide "
                  << num_PEs_per_hbm_channel << ". "
                  << "Please use graphblas::io::util_round_csr_matrix_dim. "
                  << "Exit!" <<std::endl;
        exit(EXIT_FAILURE);
    }
    assert(out_buf_len % (num_PEs_per_hbm_channel * num_hbm_channels) == 0);
    assert(vec_buf_len % num_PEs_per_hbm_channel == 0);
    CPSRMatrix<data_type, num_PEs_per_hbm_channel> cpsr_matrix;
    cpsr_matrix.skip_empty_rows = skip_empty_rows;
    cpsr_matrix.num_hbm_channels = num_hbm_channels;
    cpsr_matrix.num_row_partitions = (csr_matrix.num_rows + out_buf_len - 1) / out_buf_len;
    cpsr_matrix.num_col_partitions = (csr_matrix.num_cols + vec_buf_len - 1) / vec_buf_len;
    uint32_t num_partitions = cpsr_matrix.num_row_partitions*cpsr_matrix.num_col_partitions;
    cpsr_matrix.formatted_adj_data.resize(num_partitions * num_hbm_channels);
    cpsr_matrix.formatted_adj_indices.resize(num_partitions * num_hbm_channels);
    cpsr_matrix.formatted_adj_indptr.resize(num_partitions * num_hbm_channels);
    for (size_t j = 0; j < cpsr_matrix.num_row_partitions; j++) {
        std::vector<data_type> partitioned_adj_data[cpsr_matrix.num_col_partitions];
        std::vector<uint32_t> partitioned_adj_indices[cpsr_matrix.num_col_partitions];
        std::vector<uint32_t> partitioned_adj_indptr[cpsr_matrix.num_col_partitions];
        uint32_t num_rows = out_buf_len;
        if (j == (cpsr_matrix.num_row_partitions - 1)) {
            num_rows = csr_matrix.num_rows - (cpsr_matrix.num_row_partitions - 1) * out_buf_len;
        }
        std::vector<uint32_t> adj_indptr_slice(csr_matrix.adj_indptr.begin() + j*out_buf_len,
            csr_matrix.adj_indptr.begin() + j*out_buf_len + num_rows + 1);
        uint32_t offset = csr_matrix.adj_indptr[j * out_buf_len];
        for (auto &x : adj_indptr_slice) x -= offset;
        util_convert_csr_to_dds<data_type>(num_rows,
                                           csr_matrix.num_cols,
                                           csr_matrix.adj_data.data() + offset,
                                           csr_matrix.adj_indices.data() + offset,
                                           adj_indptr_slice.data(),
                                           vec_buf_len,
                                           partitioned_adj_data,
                                           partitioned_adj_indices,
                                           partitioned_adj_indptr);
        for (size_t i = 0; i < cpsr_matrix.num_col_partitions; i++) {
            util_pad_marker_end_of_row<data_type>(partitioned_adj_data[i],
                                                  partitioned_adj_indices[i],
                                                  partitioned_adj_indptr[i],
                                                  idx_marker,
                                                  num_hbm_channels*num_PEs_per_hbm_channel,
                                                  skip_empty_rows);
            util_pack_rows<data_type,
                           typename CPSRMatrix<data_type, num_PEs_per_hbm_channel>::packed_val_t,
                           typename CPSRMatrix<data_type, num_PEs_per_hbm_channel>::packed_idx_t>(
                partitioned_adj_data[i],
                partitioned_adj_indices[i],
                partitioned_adj_indptr[i],
                num_hbm_channels,
                num_PEs_per_hbm_channel,
                &(cpsr_matrix.formatted_adj_data[j*cpsr_matrix.num_col_partitions*num_hbm_channels
                                                 + i*num_hbm_channels]),
                &(cpsr_matrix.formatted_adj_indices[j*cpsr_matrix.num_col_partitions*num_hbm_channels
                                                    + i*num_hbm_channels]),
                &(cpsr_matrix.formatted_adj_indptr[j*cpsr_matrix.num_col_partitions*num_hbm_channels
                                                   + i*num_hbm_channels]));
        }
    }
    return cpsr_matrix;
}


//----------------------------
// Variants of CSC for SpMSpV
//----------------------------

// Fromatted CSC matrix
// we do packing, padding and partitoning when formatting the standard CSC matrix
template<typename MatrixPacketT>
struct FormattedCSCMatrix {
    /*! \brief The number of columns */
    uint32_t num_columns;
    /*! \brief The number of partitions along the row dimension */
    uint32_t num_row_partitions;
    /*! \brief The total number of packets in the matrix (used for memory allocation) */
    uint32_t num_packets_total;

    std::vector<MatrixPacketT> formatted_adj_packet;
    std::vector<graphblas::idx_t> formatted_adj_indptr;
    std::vector<graphblas::idx_t> formatted_adj_partptr;

    /*!
     * \brief get formatted packet
     */
    std::vector<MatrixPacketT, aligned_allocator<MatrixPacketT>> get_formatted_packet() {
        std::vector<MatrixPacketT, aligned_allocator<MatrixPacketT>> channel_packets;
        channel_packets.resize(this->num_packets_total);
        for (size_t i = 0; i < this->num_packets_total; i++) {
            channel_packets[i] = this->formatted_adj_packet[i];
        }
        return channel_packets;
    }

    /*!
     * \brief get a formatted indptr
     */
    std::vector<graphblas::idx_t, aligned_allocator<graphblas::idx_t>> get_formatted_indptr() {
        std::vector<graphblas::idx_t, aligned_allocator<graphblas::idx_t>> channel_indptr;
        channel_indptr.resize((this->num_columns + 1) * this->num_row_partitions);
        for (size_t i = 0; i < (this->num_columns + 1) * this->num_row_partitions; i++) {
            channel_indptr[i] = this->formatted_adj_indptr[i];
        }
        return channel_indptr;
    }

    /*!
     * \brief get a formatted partptr
     */
    std::vector<graphblas::idx_t, aligned_allocator<graphblas::idx_t>> get_formatted_partptr() {
        std::vector<graphblas::idx_t, aligned_allocator<graphblas::idx_t>> channel_partptr;
        channel_partptr.resize(this->num_row_partitions);
        for (size_t i = 0; i < this->num_row_partitions; i++) {
            channel_partptr[i] = this->formatted_adj_partptr[i];
        }
        return channel_partptr;
    }
};

/*!
 * \brief Format a standard CSC matrix.
 *
 * \tparam DataT The data type of non-zero values of the sparse matrix.
 * \tparam MatrixPacketT The packet type of the formatted matrix.
 *
 * \param csc_matrix The input matrix in CSC format.
 * \param semiring The semiring on which this matrix is represented.
 * \param pack_size The number of elements packed in one packet.
 * \param out_buf_len The length of the output buffer. Must divide (pack_size * 2).
 *
 * \return The output matrix in CPSR format.
 */

template<typename DataT, typename MatrixPacketT>
FormattedCSCMatrix<MatrixPacketT> formatCSC(CSCMatrix<DataT> const &csc_matrix,
                                            SemiringType semiring,
                                            uint32_t pack_size,
                                            uint32_t out_buf_len) {
    if (out_buf_len % (pack_size * 2) != 0) {
        std::cout << "ERROR: [formatCSC] The out_buf_len should divide (pack_size * 2) !"
                  << "  Aborting..." <<std::endl;
        exit(EXIT_FAILURE);
    }

    assert(out_buf_len % (pack_size * 2) == 0);
    FormattedCSCMatrix<MatrixPacketT> formatted_matrix;
    formatted_matrix.num_columns = csc_matrix.num_cols;
    formatted_matrix.num_row_partitions = (csc_matrix.num_rows + out_buf_len - 1) / out_buf_len;
    formatted_matrix.formatted_adj_packet.clear();
    formatted_matrix.formatted_adj_indptr.clear();
    formatted_matrix.formatted_adj_partptr.clear();

    // temporary buffers
    std::vector<std::vector<DataT>> tile_data_buf(formatted_matrix.num_row_partitions);
    std::vector<std::vector<uint32_t>> tile_idx_buf(formatted_matrix.num_row_partitions);
    // accumulative buffers
    std::vector<std::vector<uint32_t>> tile_idxptr_buf(formatted_matrix.num_row_partitions);
    std::vector<std::vector<MatrixPacketT>> tile_ditpkt_buf(formatted_matrix.num_row_partitions);

    // tile nnz counter (temporary)
    std::vector<unsigned> tile_nnz_cnt(formatted_matrix.num_row_partitions,0);

    // tile packet counter (accumulative)
    std::vector<unsigned> tile_pkt_cnt(formatted_matrix.num_row_partitions,0);

    uint32_t total_num_packets = 0;

    // add initial tile idxptr
    for (size_t t = 0; t < formatted_matrix.num_row_partitions; t++) {
      tile_idxptr_buf[t].push_back(0);
    }

    // add initial tileptr
    formatted_matrix.formatted_adj_partptr.push_back(0);

    // loop over all columns
    for (unsigned i = 0; i < csc_matrix.num_cols; i++) {
      // slice out one column
      uint32_t start = csc_matrix.adj_indptr[i];
      uint32_t end = csc_matrix.adj_indptr[i+1];
      uint32_t col_len = end - start;

      // clear temporary buffer
      for (size_t t = 0; t < formatted_matrix.num_row_partitions; t++) {
        tile_data_buf[t].clear();
        tile_idx_buf[t].clear();
        tile_nnz_cnt[t] = 0;
      }

      // loop over all rows and distribute to the corresbonding tile
      for (unsigned j = 0; j < col_len; j++) {
        unsigned dest_tile = csc_matrix.adj_indices[start + j] / out_buf_len;
        tile_data_buf[dest_tile].push_back(csc_matrix.adj_data[start + j]);
        tile_idx_buf[dest_tile].push_back(csc_matrix.adj_indices[start + j]);
        tile_nnz_cnt[dest_tile] ++;
      }

      // column padding and data packing for every tile
      for (unsigned t = 0; t < formatted_matrix.num_row_partitions; t++) {

        // padding with zero
        unsigned num_packets = (tile_nnz_cnt[t] + pack_size - 1) / pack_size;
        unsigned num_padding_zero = num_packets * pack_size - tile_nnz_cnt[t];
        for (size_t z = 0; z < num_padding_zero; z++) {
          tile_data_buf[t].push_back(semiring.zero);
          tile_idx_buf[t].push_back(0);
        }
        tile_pkt_cnt[t] += num_packets;
        total_num_packets += num_packets;

        // data packing
        for (unsigned p = 0; p < num_packets; p++) {
          MatrixPacketT dwi_packet;
          for (unsigned k = 0; k < pack_size; k++) {
            dwi_packet.vals[k] = tile_data_buf[t][k + p * pack_size];
            dwi_packet.indices[k] = tile_idx_buf[t][k + p * pack_size];
          }
          tile_ditpkt_buf[t].push_back(dwi_packet);
        }
      }

      // append tile idxptr
      for (size_t t = 0; t < formatted_matrix.num_row_partitions; t++) {
        tile_idxptr_buf[t].push_back(tile_pkt_cnt[t]);
      }
    } // repeat for every column

    // concatenate all accumulative buffers into final output
    // and create tileptr vector
    for (size_t t = 0; t < formatted_matrix.num_row_partitions; t++) {
      // ditpkt
      formatted_matrix.formatted_adj_packet.insert(formatted_matrix.formatted_adj_packet.end(),
                                                   tile_ditpkt_buf[t].begin(),
                                                   tile_ditpkt_buf[t].end());
      // idxptr
      formatted_matrix.formatted_adj_indptr.insert(formatted_matrix.formatted_adj_indptr.end(),
                                                   tile_idxptr_buf[t].begin(),
                                                   tile_idxptr_buf[t].end());
      // tileptr
      formatted_matrix.formatted_adj_partptr.push_back(tile_pkt_cnt[t] + formatted_matrix.formatted_adj_partptr.back());
    }
    std::cout << "Total #of Packets : " << formatted_matrix.formatted_adj_packet.size()  << std::endl;
    std::cout << "Total #of Tiles   : " << formatted_matrix.num_row_partitions           << std::endl;
    std::cout << "Size of idxptr    : " << formatted_matrix.formatted_adj_indptr.size()  << std::endl;
    std::cout << "Size of tileptr   : " << formatted_matrix.formatted_adj_partptr.size() << std::endl;
    formatted_matrix.num_packets_total = total_num_packets;
    return formatted_matrix;
}

} // namespace io
} // namespace graphblas

#endif // __GRAPHBLAS_IO_DATA_FORMATTER_H
