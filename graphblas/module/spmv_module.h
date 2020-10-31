#ifndef __GRAPHBLAS_SPMV_MODULE_H
#define __GRAPHBLAS_SPMV_MODULE_H

#include <cstdint>
#include <vector>
#include <fstream>
#include <chrono>

#include "xcl2.hpp"

#include "../global.h"
#include "../io/data_loader.h"
#include "../io/data_formatter.h"
#include "./base_module.h"

using graphblas::io::CSRMatrix;
using graphblas::io::CPSRMatrix;
using graphblas::io::csr2cpsr;

namespace graphblas {
namespace module {

template<typename matrix_data_t, typename vector_data_t>
class SpMVModule : public BaseModule {
private:
    /*! \brief The mask type */
    graphblas::MaskType mask_type_;
    /*! \brief The semiring */
    SemiRingType semiring_;
    /*! \brief The number of channels of the kernel */
    uint32_t num_channels_;
    /*! \brief The length of output buffer of the kernel */
    uint32_t out_buf_len_;
    /*! \brief The number of row partitions */
    uint32_t num_row_partitions_;
    /*! \brief The length of vector buffer of the kernel */
    uint32_t vec_buf_len_;
    /*! \brief The number of column partitions */
    uint32_t num_col_partitions_;

    using val_t = vector_data_t;
    using packed_val_t = typename CPSRMatrix<val_t, graphblas::pack_size>::packed_val_t;
    using packed_idx_t = typename CPSRMatrix<val_t, graphblas::pack_size>::packed_idx_t;
    using mat_pkt_t = struct {packed_idx_t indices; packed_val_t vals;};
    using partition_indptr_t = struct {graphblas::idx_t start; packed_idx_t nnz;};

    using aligned_idx_t = std::vector<graphblas::idx_t, aligned_allocator<graphblas::idx_t>>;
    using aligned_dense_vec_t = std::vector<val_t, aligned_allocator<val_t>>;
    using aligned_packed_idx_t = std::vector<packed_idx_t, aligned_allocator<packed_idx_t>>;
    using aligned_packed_val_t = std::vector<packed_val_t, aligned_allocator<packed_val_t>>;
    using aligned_mat_pkt_t = std::vector<mat_pkt_t, aligned_allocator<mat_pkt_t>>;
    using aligned_partition_indptr_t = std::vector<partition_indptr_t, aligned_allocator<partition_indptr_t>>;

    /*! \brief Matrix packets (indices and vals) + partition indptr for each channel */
    std::vector<aligned_mat_pkt_t> channel_packets_;
    /*! \brief Internal copy of the dense vector */
    aligned_dense_vec_t vector_;
    /*! \brief Internal copy of mask */
    aligned_dense_vec_t mask_;
    /*! \brief The argument index of mask to be used in setArg */
    uint32_t arg_idx_mask_;
    /*! \brief The kernel results */
    aligned_dense_vec_t results_;
    /*! \brief The sparse matrix using float data type in CSR format */
    CSRMatrix<float> csr_matrix_float_;
    /*! \brief The sparse matrix in CSR format */
    CSRMatrix<val_t> csr_matrix_;
    /*! \brief The sparse matrix in CPSR format */
    CPSRMatrix<val_t, graphblas::pack_size> cpsr_matrix_;

public:
    // Device buffers
    std::vector<cl::Buffer> channel_packets_buf;
    cl::Buffer vector_buf;
    cl::Buffer mask_buf;
    cl::Buffer results_buf;

private:
    /*!
     * \brief The matrix data type should be the same as the vector data type.
     */
    void _check_data_type();

    /*!
     * \brief Get the kernel configuration.
     * \param num_channels The number of channels.
     * \param out_buf_len The length of output buffer.
     * \param vec_buf_len The length of vector buffer.
     */
    void _get_kernel_config(uint32_t num_channels,
                            uint32_t out_buf_len,
                            uint32_t vec_buf_len);

public:
    SpMVModule(uint32_t num_channels,
               uint32_t out_buf_len,
               uint32_t vec_buf_len) : BaseModule("kernel_spmv") {
        this->_check_data_type();
        this->_get_kernel_config(num_channels, out_buf_len, vec_buf_len);
    }

    /*!
     * \brief Set the semiring type.
     * \param semiring The semiring type.
     */
    void set_semiring(graphblas::SemiRingType semiring) {
        this->semiring_ = semiring;
    }

    /*!
     * \brief Set the mask type.
     * \param mask_type The mask type.
     */
    void set_mask_type(graphblas::MaskType mask_type) {
        this->mask_type_ = mask_type;
    }

    /*!
     * \brief Get the number of rows of the sparse matrix.
     * \return The number of rows.
     */
    uint32_t get_num_rows() {
        return this->csr_matrix_.num_rows;
    }

    /*!
     * \brief Get the number of columns of the sparse matrix.
     * \return The number of columns.
     */
    uint32_t get_num_cols() {
        return this->csr_matrix_.num_cols;
    }

    /*!
     * \brief Get the number of non-zeros of the sparse matrix.
     * \return The number of non-zeros.
     */
    uint32_t get_nnz() {
        return this->csr_matrix_.adj_indptr[this->csr_matrix_.num_rows];
    }

    /*!
     * \brief Load a csr matrix and format the csr matrix.
     *        The csr matrix should have float data type.
     *        Data type conversion, if required, is handled internally.
     * \param csr_matrix_float The csr matrix using float data type.
     */
    void load_and_format_matrix(CSRMatrix<float> const &csr_matrix_float);

    /*!
     * \brief Send the formatted matrix from host to device.
     */
    void send_matrix_host_to_device();

    /*!
     * \brief Send the dense vector from host to device.
     */
    void send_vector_host_to_device(aligned_dense_vec_t &vector);

    /*!
     * \brief Send the mask from host to device.
     */
    void send_mask_host_to_device(aligned_dense_vec_t &mask);

    // uint32_t count_elements_skip_empty_rows();

    /*!
     * \brief Run the module.
     */
    void run();

    /*!
     * \brief Send the dense vector from device to host.
     */
    aligned_dense_vec_t send_vector_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->vector_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->vector_;
    }

    /*!
     * \brief Send the mask from device to host.
     * \return The mask.
     */
    aligned_dense_vec_t send_mask_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->mask_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->mask_;
    }

    /*!
     * \brief Send the results from device to host.
     * \return The results.
     */
    aligned_dense_vec_t send_results_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->results_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->results_;
    }

    /*!
     * \brief Compute reference results.
     * \param vector The dense vector.
     * \return The reference results.
     */
    graphblas::aligned_dense_float_vec_t
    compute_reference_results(graphblas::aligned_dense_float_vec_t &vector);

    /*!
     * \brief Compute reference results.
     * \param vector The dense vector.
     * \param mask The mask.
     * \return The reference results.
     */
    graphblas::aligned_dense_float_vec_t
    compute_reference_results(graphblas::aligned_dense_float_vec_t &vector,
                              graphblas::aligned_dense_float_vec_t &mask);

    void generate_kernel_header() override;

    void generate_kernel_ini() override;
};


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::_check_data_type() {
    assert((std::is_same<matrix_data_t, vector_data_t>::value));
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::_get_kernel_config(uint32_t num_channels,
                                                                  uint32_t out_buf_len,
                                                                  uint32_t vec_buf_len) {
    this->num_channels_ = num_channels;
    this->out_buf_len_ = out_buf_len;
    this->vec_buf_len_ = vec_buf_len;
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::generate_kernel_header() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream header(graphblas::proj_folder_name + "/" + this->kernel_name_ + ".h", std::ios_base::app);
    header << "const unsigned OUT_BUF_LEN = " << this->out_buf_len_ << ";" << std::endl;
    header << "const unsigned VEC_BUF_LEN = " << this->vec_buf_len_ << ";" << std::endl;
    header << "const unsigned NUM_HBM_CHANNEL = " << this->num_channels_ << ";" << std::endl;
    header << "#define NUM_HBM_CHANNEL " << this->num_channels_ << std::endl;
    header << "const unsigned NUM_PE_TOTAL = "
           << this->num_channels_ * graphblas::pack_size << ";" << std::endl;
    header.close();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::generate_kernel_ini() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream ini(graphblas::proj_folder_name + "/" + this->kernel_name_ + ".ini");
    ini << "[connectivity]" << std::endl;
    // HBM
    for (size_t hbm_idx = 0; hbm_idx < this->num_channels_; hbm_idx++) {
        ini << "sp=kernel_spmv_1.channel_" << hbm_idx << "_matrix:HBM[" << hbm_idx << "]" << std::endl;
    }
    ini << "sp=kernel_spmv_1.vector:DDR[0]" << std::endl;
    ini << "sp=kernel_spmv_1.mask:DDR[0]" << std::endl;
    ini << "sp=kernel_spmv_1.out:DDR[0]" << std::endl;
    ini.close();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::load_and_format_matrix(
        CSRMatrix<float> const &csr_matrix_float) {
    this->csr_matrix_float_ = csr_matrix_float;
    this->csr_matrix_ = graphblas::io::csr_matrix_convert_from_float<val_t>(csr_matrix_float);
    this->num_row_partitions_ = (this->csr_matrix_.num_rows + this->out_buf_len_ - 1) / this->out_buf_len_;
    this->num_col_partitions_ = (this->csr_matrix_.num_cols + this->vec_buf_len_ - 1) / this->vec_buf_len_;
    size_t num_partitions = this->num_row_partitions_ * this->num_col_partitions_;
    val_t val_marker = 0;
    this->cpsr_matrix_ = csr2cpsr<val_t, graphblas::pack_size>(
        this->csr_matrix_,
        val_marker,
        graphblas::idx_marker,
        this->out_buf_len_,
        this->vec_buf_len_,
        this->num_channels_);
    std::vector<aligned_partition_indptr_t> channel_partition_indptr(this->num_channels_);
    std::vector<aligned_packed_idx_t> channel_indices(this->num_channels_);
    std::vector<aligned_packed_val_t> channel_vals(this->num_channels_);
    this->channel_packets_.resize(this->num_channels_);
    for (size_t c = 0; c < this->num_channels_; c++) {
        channel_partition_indptr[c].resize(num_partitions);
        channel_partition_indptr[c][0].start = 0;
    }
    for (size_t c = 0; c < this->num_channels_; c++) {
        for (size_t j = 0; j < this->num_row_partitions_; j++) {
            for (size_t i = 0; i < this->num_col_partitions_; i++) {
                auto indices_partition = this->cpsr_matrix_.get_packed_indices(j, i, c);
                channel_indices[c].insert(channel_indices[c].end(),
                    indices_partition.begin(), indices_partition.end());
                auto vals_partition = this->cpsr_matrix_.get_packed_data(j, i, c);
                channel_vals[c].insert(channel_vals[c].end(),
                    vals_partition.begin(), vals_partition.end());
                assert(indices_partition.size() == vals_partition.size());
                auto indptr_partition = this->cpsr_matrix_.get_packed_indptr(j, i, c);
                if (!((j == (this->num_row_partitions_ - 1)) && (i == (this->num_col_partitions_ - 1)))) {
                    channel_partition_indptr[c][j*this->num_col_partitions_ + i + 1].start =
                        channel_partition_indptr[c][j*this->num_col_partitions_ + i].start
                        + indices_partition.size();
                }
                channel_partition_indptr[c][j*this->num_col_partitions_ + i].nnz = indptr_partition.back();
            }
        }
        assert(channel_indices[c].size() == channel_vals[c].size());
        this->channel_packets_[c].resize(num_partitions + channel_indices[c].size());
        // partition indptr
        for (size_t i = 0; i < num_partitions; i++) {
            this->channel_packets_[c][i].indices.data[0] = channel_partition_indptr[c][i].start;
            for (size_t k = 0; k < graphblas::pack_size; k++) {
                this->channel_packets_[c][i].vals.data[k] = (val_t)channel_partition_indptr[c][i].nnz.data[k];
            }
        }
        // matrix indices and vals
        for (size_t i = 0; i < channel_indices[c].size(); i++) {
            this->channel_packets_[c][i + num_partitions].indices = channel_indices[c][i];
            this->channel_packets_[c][i + num_partitions].vals = channel_vals[c][i];
        }
    }
    this->vector_.resize(this->csr_matrix_.num_cols);
    this->results_.resize(this->csr_matrix_.num_rows);
    std::fill(this->results_.begin(), this->results_.end(), 0);
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_matrix_host_to_device() {
    cl_int err;
    // Handle channel_packets
    cl_mem_ext_ptr_t channel_packets_ext[this->num_channels_];
    this->channel_packets_buf.resize(this->num_channels_);
    for (size_t c = 0; c < this->num_channels_; c++) {
        channel_packets_ext[c].obj = this->channel_packets_[c].data();
        channel_packets_ext[c].param = 0;
        channel_packets_ext[c].flags = graphblas::HBM[c];
        size_t channel_packets_size = sizeof(mat_pkt_t) * this->channel_packets_[c].size();
        if (channel_packets_size >= 256 * 1000 * 1000) {
            std::cout << "The capcity of one HBM channel is 256 MB" << std::endl;
            exit(EXIT_FAILURE);
        }
        OCL_CHECK(err, this->channel_packets_buf[c] = cl::Buffer(this->context_,
            CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
            channel_packets_size,
            &channel_packets_ext[c],
            &err));
    }
    // Handle results
    cl_mem_ext_ptr_t results_ext;
    results_ext.obj = this->results_.data();
    results_ext.param = 0;
    results_ext.flags = graphblas::DDR[0];
    OCL_CHECK(err, this->results_buf = cl::Buffer(this->context_,
        CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(val_t) * this->csr_matrix_.num_rows,
        &results_ext,
        &err));
    // Set arguments
    size_t arg_idx = 1;  // the first argument (arg_idx = 0) is the dense vector
    for (size_t c = 0; c < this->num_channels_; c++) {
        OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->channel_packets_buf[c]));
    }
    this->arg_idx_mask_ = arg_idx;  // mask is right before results
    arg_idx++;
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->results_buf));
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->csr_matrix_.num_rows));
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->csr_matrix_.num_cols));
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, (char)this->semiring_));
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, (char)this->mask_type_));
    // Send data to device
    for (size_t c = 0; c < this->num_channels_; c++) {
        OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects(
            {this->channel_packets_buf[c]}, 0 /* 0 means from host to device */ ));
    }
    this->command_queue_.finish();
}


// template<typename matrix_data_t, typename vector_data_t>
// uint32_t SpMVModule<matrix_data_t, vector_data_t>::count_elements_skip_empty_rows() {
//     uint32_t count = 0;
//     for (size_t c = 0; c < this->num_channels_; c++) {
//         auto packets = this->channel_packets_[c];
//         uint32_t num_elements_skip_empty_rows[graphblas::pack_size];
//         for (size_t i = 0; i < graphblas::pack_size; i++) {
//             num_elements_skip_empty_rows[i] = packets.size();
//             for (size_t j = 0; j < packets.size() - 1; j++) {
//                 if ((packets[j].indices.data[i] == idx_marker) && (packets[j+1].indices.data[i] == idx_marker)) {
//                     num_elements_skip_empty_rows[i]--;
//                 }
//             }
//         }
//         count += ((graphblas::pack_size) * (*std::max_element(num_elements_skip_empty_rows,
//             num_elements_skip_empty_rows + graphblas::pack_size)));
//     }
//     return count;
// }


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_vector_host_to_device(aligned_dense_vec_t &vector) {
    this->vector_.assign(vector.begin(), vector.end());
    cl_mem_ext_ptr_t vector_ext;
    vector_ext.obj = this->vector_.data();
    vector_ext.param = 0;
    vector_ext.flags = graphblas::DDR[0];
    cl_int err;
    OCL_CHECK(err, this->vector_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(val_t) * this->csr_matrix_.num_cols,
                &vector_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(0, this->vector_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->vector_buf}, 0));
    this->command_queue_.finish();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_mask_host_to_device(aligned_dense_vec_t &mask) {
    this->mask_.assign(mask.begin(), mask.end());
    cl_mem_ext_ptr_t mask_ext;
    mask_ext.obj = this->mask_.data();
    mask_ext.param = 0;
    mask_ext.flags = graphblas::DDR[0];
    cl_int err;
    OCL_CHECK(err, this->mask_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(val_t) * this->csr_matrix_.num_rows,
                &mask_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(this->arg_idx_mask_, this->mask_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->mask_buf}, 0));
    this->command_queue_.finish();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::run() {
    cl_int err;
    OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->kernel_));
    this->command_queue_.finish();
}


#define SPMV(stmt)                                                              { \
for (size_t row_idx = 0; row_idx < this->csr_matrix_float_.num_rows; row_idx++) { \
    idx_t start = this->csr_matrix_float_.adj_indptr[row_idx];                    \
    idx_t end = this->csr_matrix_float_.adj_indptr[row_idx + 1];                  \
    for (size_t i = start; i < end; i++) {                                        \
        idx_t idx = this->csr_matrix_float_.adj_indices[i];                       \
        stmt;                                                                     \
    }                                                                             \
}                                                                               } \

template<typename matrix_data_t, typename vector_data_t>
graphblas::aligned_dense_float_vec_t
SpMVModule<matrix_data_t, vector_data_t>::compute_reference_results(aligned_dense_float_vec_t &vector) {
    aligned_dense_float_vec_t reference_results(this->csr_matrix_.num_rows);
    std::fill(reference_results.begin(), reference_results.end(), 0);
    switch (this->semiring_) {
        case graphblas::kMulAdd:
            SPMV(reference_results[row_idx] += this->csr_matrix_float_.adj_data[i] * vector[idx]);
            break;
        case graphblas::kLogicalAndOr:
            SPMV(reference_results[row_idx] = reference_results[row_idx]
                || (this->csr_matrix_float_.adj_data[i] && vector[idx]));
            break;
        default:
            std::cerr << "Invalid semiring" << std::endl;
            break;
    }
    return reference_results;
}


template<typename matrix_data_t, typename vector_data_t>
graphblas::aligned_dense_float_vec_t SpMVModule<matrix_data_t, vector_data_t>::compute_reference_results(
        graphblas::aligned_dense_float_vec_t &vector,
        graphblas::aligned_dense_float_vec_t &mask) {
    graphblas::aligned_dense_float_vec_t reference_results = this->compute_reference_results(vector);
    if (this->mask_type_ == graphblas::kMaskWriteToZero) {
        for (size_t i = 0; i < this->csr_matrix_.num_rows; i++) {
            if (mask[i] != 0) {
                reference_results[i] = 0;
            }
        }
    } else {
        for (size_t i = 0; i < this->csr_matrix_.num_rows; i++) {
            if (mask[i] == 0) {
                reference_results[i] = 0;
            }
        }
    }
    return reference_results;
}


}  // namespace module
}  // namespace graphblas

#endif  // __GRAPHBLAS_SPMV_MODULE_H
