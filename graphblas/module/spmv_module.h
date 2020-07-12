#ifndef __GRAPHBLAS_SPMV_MODULE_H
#define __GRAPHBLAS_SPMV_MODULE_H

#include <cstdint>
#include <vector>
#include <type_traits>

#include "xcl2.hpp"
#include "graph_partitioning.h"

#include "../base.h"


namespace graphblas {
namespace module {

template<typename matrix_data_t, typename vector_data_t>
class SpMVModule {
private:
    /*! \brief The number of rows of the original sparse matrix */
    uint32_t num_rows_;
    /*! \brief The number of columns of the original sparse matrix */
    uint32_t num_cols_;
    /*! \brief The index pointers (CSR) of the original sparse matrix */
    std::vector<index_t> indptr_;
    /*! \brief The column indices (CSR) of the original sparse matrix */
    std::vector<index_t> indices_;
    /*! \brief The non-zero values (CSR) of the original sparse matrix */
    std::vector<vector_data_t> vals_;
    /*! \brief Vals using float data type */
    std::vector<float> vals_float_;
    /*! \brief The number of non-zeros of the original sparse matrix */
    uint32_t nnz_;

    /*! \brief Whether the graph is weighted */
    bool graph_is_weighed_;
    /*! \brief The semiring */
    SemiRingType semiring_;
    /*! \brief The number of channels of the kernel */
    uint32_t num_channels_;
    /*! \brief The length of vector buffer of the kernel */
    uint32_t vector_buffer_len_;
    /*! \brief The number of column partitions */
    uint32_t num_col_partitions_;

    /*! \brief Type of packed data */
    using packed_data_t = struct {vector_data_t data[graphblas::pack_size];};

    /*! \brief Indptr of the partitions for each channel */
    std::vector<std::vector<index_t, aligned_allocator<index_t>>> channel_partition_indptr_;
    /*! \brief Indices for each channel */
    std::vector<std::vector<packed_index_t, aligned_allocator<packed_index_t>>> channel_indices_;
    /*! \brief Vals for each channel */
    std::vector<std::vector<packed_data_t, aligned_allocator<packed_index_t>>> channel_vals_;
    /*! \brief The dense vector */
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> vector_;
    /*! \brief The kernel results */
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> kernel_results_;

    // OpenCL runtime
    cl::Device device_;
    cl::Context context_;
    cl::Kernel kernel_;
    cl::CommandQueue command_queue_;

    // Device buffers
    std::vector<cl::Buffer> channel_partition_indptr_buf_;
    std::vector<cl::Buffer> channel_indices_buf_;
    std::vector<cl::Buffer> channel_vals_buf_;
    cl::Buffer vector_buf_;
    cl::Buffer kernel_results_buf_;

private:
    /*!
     * \brief Check whether the graph is weighted. For weighted graphs, it requires that
     *        the matrix data type is the same as the vector data type.
     */
    void _check_data_type();

    /*!
     * \brief Get the kernel Configuration.
     * \param semiring The semiring.
     * \param num_channels The number of channels.
     * \param vector_buffer_len The length of vector buffer.
     */
    void _get_kernel_config(SemiRingType semiring, uint32_t num_channels, uint32_t vector_buffer_len);

    /*!
     * \brief Load a sparse matrix from a scipy sparse npz file and format the sparse matrix.
     *        The sparse matrix should have float data type.
     *        Data type conversion, if required, is handled internally.
     * \param csr_float_npz_path The file path.
     */
    void _load_and_format_data(std::string csr_float_npz_path);

    /*!
     * \brief Send the formatted sparse matrix to FPGA.
     */
    void _send_data_to_FPGA();

    /*!
     * \brief Load the xclbin file and set up runtime.
     * \param xclbin_file_path The xclbin file path.
     * \param kernel_name The kernel name.
     */
    void _set_up_runtime(std::string xclbin_file_path, std::string kernel_name);

public:
    SpMVModule(std::string csr_float_npz_path,
               SemiRingType semiring,
               uint32_t num_channels,
               uint32_t vector_buffer_len,
               std::string xclbin_file_path,
               std::string kernel_name) {
        this->_check_data_type();
        this->_get_kernel_config(semiring, num_channels, vector_buffer_len);
        this->_load_and_format_data(csr_float_npz_path);
        this->device_ = graphblas::find_device();
        this->context_ = cl::Context(this->device_, NULL, NULL, NULL);
        this->_send_data_to_FPGA();
        this->_set_up_runtime(xclbin_file_path, kernel_name);
    }

    using aligned_vector_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;
    /*!
     * \brief Run the module.
     * \param vector The input vector.
     * \return The kernel results.
     */
    aligned_vector_t run(aligned_vector_t &vector) {
        cl_mem_ext_ptr_t vector_ext;
        vector_ext.obj = vector.data();
        vector_ext.param = 0;
        vector_ext.flags = 0;
        cl_int err;
        OCL_CHECK(err, this->vector_buf_ = cl::Buffer(this->context_,
                  CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                  sizeof(vector_data_t) * this->num_cols_,
                  &vector_ext,
                  &err));
        OCL_CHECK(err, err = this->kernel_.setArg(0, this->vector_buf_));
        OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->vector_buf_}, 0));
        OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->kernel_));
        this->command_queue_.finish();
        OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->kernel_results_buf_},
            CL_MIGRATE_MEM_OBJECT_HOST));
        this->command_queue_.finish();
        return this->kernel_results_;
    }

    using aligned_float_t = std::vector<float, aligned_allocator<float>>;
    /*!
     * \brief Compute reference results.
     * \param vector The input vector.
     * \return The reference results.
     */
    aligned_float_t compute_reference_results(aligned_float_t &vector) {
        aligned_float_t reference_results(this->num_rows_);
        std::fill(reference_results.begin(), reference_results.end(), 0);
        if (this->graph_is_weighed_) {
            for (size_t row_idx = 0; row_idx < this->num_rows_; row_idx++) {
                index_t start = this->indptr_[row_idx];
                index_t end = this->indptr_[row_idx + 1];
                for (size_t i = start; i < end; i++) {
                    index_t index = this->indices_[i];
                    reference_results[row_idx] += this->vals_float_[i] * vector[index];
                }
            }
        } else {
            for (size_t row_idx = 0; row_idx < this->num_rows_; row_idx++) {
                index_t start = this->indptr_[row_idx];
                index_t end = this->indptr_[row_idx + 1];
                for (size_t i = start; i < end; i++) {
                    index_t index = this->indices_[i];
                    reference_results[row_idx] += vector[index];
                }
            }
        }
        return reference_results;
    }

    // float measure_average_time(uint32_t num_runs) {

    // }

};


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::_check_data_type() {
    if (std::is_same<matrix_data_t, bool>::value) {
        this->graph_is_weighed_ = false;
    } else {
        this->graph_is_weighed_ = true;
    }
    if (!std::is_same<matrix_data_t, bool>::value) {
        assert((std::is_same<matrix_data_t, vector_data_t>::value));
    }
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::_get_kernel_config(SemiRingType semiring,
                                                                  uint32_t num_channels,
                                                                  uint32_t vector_buffer_len) {
    this->semiring_ = semiring;
    this->num_channels_ = num_channels;
    this->vector_buffer_len_ = vector_buffer_len;
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::_load_and_format_data(std::string csr_float_npz_path) {
    SpMVDataFormatter<vector_data_t, graphblas::pack_size, packed_data_t, graphblas::packed_index_t>
        formatter(csr_float_npz_path);
    this->num_rows_ = formatter.get_num_rows();
    this->num_cols_ = formatter.get_num_cols();
    this->indptr_ = formatter.get_indptr();
    this->indices_ = formatter.get_indices();
    this->vals_ = formatter.get_data();
    this->vals_float_ = formatter.get_data_float();
    this->nnz_ = this->indptr_[this->num_rows_];
    this->num_col_partitions_ = (this->num_cols_ + this->vector_buffer_len_ - 1)
        / this->vector_buffer_len_;
    matrix_data_t val_marker = 0;
    formatter.format_pad_marker_end_of_row(this->vector_buffer_len_,
                                           this->num_channels_,
                                           val_marker,
                                           graphblas::idx_marker);
    this->channel_indices_.resize(this->num_channels_);
    this->channel_partition_indptr_.resize(this->num_channels_);
    for (size_t c = 0; c < this->num_channels_; c++) {
        this->channel_partition_indptr_[c].resize(this->num_col_partitions_ + 1);
        this->channel_partition_indptr_[c][0] = 0;
    }
    if (this->graph_is_weighed_) {
        this->channel_vals_.resize(this->num_channels_);
    }
    for (size_t c = 0; c < this->num_channels_; c++) {
        for (size_t i = 0; i < this->num_col_partitions_; i++) {
            auto indices_partition = formatter.get_packed_indices(i, c);
            this->channel_partition_indptr_[c][i + 1] =
                this->channel_partition_indptr_[c][i] + indices_partition.size();
            this->channel_indices_[c].insert(this->channel_indices_[c].end(),
                indices_partition.begin(), indices_partition.end());
            if (this->graph_is_weighed_) {
                auto vals_partition = formatter.get_packed_data(i, c);
                this->channel_vals_[c].insert(this->channel_vals_[c].end(),
                    vals_partition.begin(), vals_partition.end());
            }
        }
    }
    this->vector_.resize(this->num_cols_);
    this->kernel_results_.resize(this->num_rows_);
    std::fill(this->kernel_results_.begin(), this->kernel_results_.end(), 0);
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::_send_data_to_FPGA() {
    cl_int err;
    // Handle channel_partition_indptr and channel_indices
    cl_mem_ext_ptr_t channel_partition_indptr_ext[this->num_channels_];
    cl_mem_ext_ptr_t channel_indices_ext[this->num_channels_];
    this->channel_partition_indptr_buf_.resize(this->num_channels_);
    this->channel_indices_buf_.resize(this->num_channels_);
    for (size_t c = 0; c < this->num_channels_; c++) {
        channel_partition_indptr_ext[c].obj = this->channel_partition_indptr_[c].data();
        channel_partition_indptr_ext[c].param = 0;
        channel_partition_indptr_ext[c].flags = 0;
        channel_indices_ext[c].obj = this->channel_indices_[c].data();
        channel_indices_ext[c].param = 0;
        channel_indices_ext[c].flags = graphblas::HBM[c];
        OCL_CHECK(err, this->channel_partition_indptr_buf_[c] = cl::Buffer(this->context_,
            CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
            sizeof(index_t) * (this->num_col_partitions_ + 1),
            &channel_partition_indptr_ext[c],
            &err));
        OCL_CHECK(err, this->channel_indices_buf_[c] = cl::Buffer(this->context_,
            CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
            sizeof(graphblas::packed_index_t) * this->channel_indices_[c].size(),
            &channel_indices_ext[c],
            &err));
    }
    // Handle channel_vals for weighted graphs
    if (this->graph_is_weighed_) {
        cl_mem_ext_ptr_t channel_vals_ext[this->num_channels_];
        for (size_t c = 0; c < this->num_channels_; c++) {
            channel_vals_ext[c].obj = this->channel_vals_[c].data();
            channel_vals_ext[c].param = 0;
            channel_vals_ext[c].flags = graphblas::HBM[this->num_channels_ + c];
            OCL_CHECK(err, this->channel_vals_buf_[c] = cl::Buffer(this->context_,
                CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(packed_data_t) * this->channel_vals_[c].size(),
                &channel_vals_ext[c],
                &err));
        }
    }
    // Handle kernel_results
    cl_mem_ext_ptr_t kernel_results_ext;
    kernel_results_ext.obj = this->kernel_results_.data();
    kernel_results_ext.param = 0;
    kernel_results_ext.flags = 0;
    OCL_CHECK(err, this->kernel_results_buf_ = cl::Buffer(this->context_,
        CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(vector_data_t) * this->num_rows_,
        &kernel_results_ext,
        &err));
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::_set_up_runtime(std::string xclbin_file_path,
                                                               std::string kernel_name) {
    // Load xclbin
    cl_int err;
    auto file_buf = xcl::read_binary_file(xclbin_file_path);
    cl::Program::Binaries binaries{{file_buf.data(), file_buf.size()}};
    cl::Program program(this->context_, {this->device_}, binaries, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file\n";
    } else {
        std::cout << "Successfully programmed device with xclbin file\n";
    }
    OCL_CHECK(err, this->kernel_ = cl::Kernel(program, kernel_name.c_str(), &err));
    // Set arguments
    size_t arg_idx = 1; // the first argument (arg_idx = 0) is the dense vector
    for (size_t c = 0; c < this->num_channels_; c++) {
        OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->channel_partition_indptr_buf_[c]));
        OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->channel_indices_buf_[c]));
        if (this->graph_is_weighed_) {
            OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->channel_vals_buf_[c]));
        }
    }
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->kernel_results_buf_));
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->num_rows_));
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->num_cols_));
    uint32_t num_times = 1;
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx, num_times));
    // Create command queue
    OCL_CHECK(err, this->command_queue_ = cl::CommandQueue(this->context_,
                                                           this->device_,
                                                           CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
                                                           &err));
    // Copy data to device global memory
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->channel_partition_indptr_buf_[0],
                                                                        this->channel_indices_buf_[0],
                                                                        this->channel_partition_indptr_buf_[1],
                                                                        this->channel_indices_buf_[1]}, 0 /* 0 means from host*/));
    this->command_queue_.finish();
}

} // namespace module
} // namespace graphblas

#endif // __GRAPHBLAS_SPMV_MODULE_H
