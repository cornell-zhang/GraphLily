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
using graphblas::io::SpMVDataFormatter;


namespace graphblas {
namespace module {

template<typename matrix_data_t, typename vector_data_t>
class SpMVModule : public BaseModule {
private:
    /*! \brief Whether the graph is weighted */
    bool graph_is_weighed_;
    /*! \brief Whether the kernel uses mask */
    bool use_mask_;
    /*! \brief The mask type */
    graphblas::MaskType mask_type_;
    /*! \brief The semiring */
    SemiRingType semiring_;
    /*! \brief The number of channels of the kernel */
    uint32_t num_channels_;
    /*! \brief The length of output buffer of the kernel */
    uint32_t out_buffer_len_;
    /*! \brief The number of row partitions */
    uint32_t num_row_partitions_;
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
    /*! \brief Internal copy of the dense vector */
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> vector_;
    /*! \brief Internal copy of mask */
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> mask_;
    /*! \brief The argument index of mask to be used in setArg */
    uint32_t arg_idx_mask_;
    /*! \brief The kernel results */
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> results_;

    // String representation of the data type
    std::string matrix_data_t_str_;
    std::string vector_data_t_str_;

    // Use float to work around the issue with std::vector<bool>
    using matrix_data_t_internal_ = typename std::conditional<std::is_same<matrix_data_t, bool>::value,
                                                              float,
                                                              matrix_data_t>::type;

    /*! \brief The sparse matrix using float data type*/
    CSRMatrix<float> csr_matrix_float_;
    /*! \brief The sparse matrix */
    CSRMatrix<matrix_data_t_internal_> csr_matrix_;

public:
    // Device buffers
    std::vector<cl::Buffer> channel_partition_indptr_buf;
    std::vector<cl::Buffer> channel_indices_buf;
    std::vector<cl::Buffer> channel_vals_buf;
    cl::Buffer vector_buf;
    cl::Buffer mask_buf;
    cl::Buffer results_buf;

private:
    /*!
     * \brief Check whether the graph is weighted. For weighted graphs, it requires that
     *        the matrix data type is the same as the vector data type.
     */
    void _check_data_type();

    /*!
     * \brief Get the kernel configuration.
     * \param semiring The semiring.
     * \param num_channels The number of channels.
     * \param out_buffer_len The length of output buffer.
     * \param vector_buffer_len The length of vector buffer.
     */
    void _get_kernel_config(SemiRingType semiring,
                            uint32_t num_channels,
                            uint32_t out_buffer_len,
                            uint32_t vector_buffer_len);

public:
    SpMVModule(SemiRingType semiring,
               uint32_t num_channels,
               uint32_t out_buffer_len,
               uint32_t vector_buffer_len) : BaseModule("kernel_spmv") {
        this->_check_data_type();
        this->_get_kernel_config(semiring, num_channels, out_buffer_len, vector_buffer_len);
    }

    /*!
     * \brief Set the mask type.
     * \param mask_type The mask type.
     */
    void set_mask_type(graphblas::MaskType mask_type) {
        if (mask_type == graphblas::kNoMask) {
            this->use_mask_ = false;
        } else {
            this->use_mask_ = true;
            this->mask_type_ = mask_type;
        }
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

    using aligned_vector_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;

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
    void send_vector_host_to_device(aligned_vector_t &vector);

    /*!
     * \brief Send the mask from host to device.
     */
    void send_mask_host_to_device(aligned_vector_t &mask);

    /*!
     * \brief Run the module.
     */
    void run();

    /*!
     * \brief Send the dense vector from device to host.
     */
    aligned_vector_t send_vector_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->vector_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->vector_;
    }

    /*!
     * \brief Send the mask from device to host.
     * \return The mask.
     */
    aligned_vector_t send_mask_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->mask_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->mask_;
    }

    /*!
     * \brief Send the results from device to host.
     * \return The results.
     */
    aligned_vector_t send_results_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->results_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->results_;
    }

    /*!
     * \brief Compute reference results.
     * \param vector The dense vector.
     * \return The reference results.
     */
    graphblas::aligned_float_t compute_reference_results(graphblas::aligned_float_t &vector);

    /*!
     * \brief Compute reference results.
     * \param vector The dense vector.
     * \param mask The mask.
     * \return The reference results.
     */
    graphblas::aligned_float_t compute_reference_results(graphblas::aligned_float_t &vector,
                                                         graphblas::aligned_float_t &mask);

    void generate_kernel_header() override;

    void generate_kernel_ini() override;

    void link_kernel_code() override;
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
    this->matrix_data_t_str_ = graphblas::dtype_to_str<matrix_data_t>();
    this->vector_data_t_str_ = graphblas::dtype_to_str<vector_data_t>();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::_get_kernel_config(SemiRingType semiring,
                                                                  uint32_t num_channels,
                                                                  uint32_t out_buffer_len,
                                                                  uint32_t vector_buffer_len) {
    this->semiring_ = semiring;
    this->num_channels_ = num_channels;
    this->out_buffer_len_ = out_buffer_len;
    this->vector_buffer_len_ = vector_buffer_len;
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::generate_kernel_header() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream header(graphblas::proj_folder_name + "/" + this->kernel_name_ + ".h");
    // Kernel configuration
    header << "const unsigned int VECTOR_PACK_SIZE = " << graphblas::pack_size << ";" << std::endl;
    header << "const unsigned int NUM_PE_PER_HBM_CHANNEL = VECTOR_PACK_SIZE" << ";" << std::endl;
    header << "const unsigned int NUM_HBM_CHANNEL = " <<  this->num_channels_ << ";" << std::endl;
    header << "const unsigned int NUM_PE_TOTAL = NUM_PE_PER_HBM_CHANNEL*NUM_HBM_CHANNEL" << ";" << std::endl;
    header << "const unsigned int NUM_PORT_PER_BANK = 2;" << std::endl;
    header << "const unsigned int NUM_BANK_PER_HBM_CHANNEL = NUM_PE_PER_HBM_CHANNEL/NUM_PORT_PER_BANK"
           << ";" << std::endl;
    header << "const unsigned int OUT_BUFFER_LEN = " << this->out_buffer_len_ << ";" << std::endl;
    header << "const unsigned int VECTOR_BUFFER_LEN = " << this->vector_buffer_len_ << ";" << std::endl;
    // Data types
    header << "typedef " << this->vector_data_t_str_ << " VECTOR_T;" << std::endl;
    header << "typedef struct {VECTOR_T data[VECTOR_PACK_SIZE];}" << " PACKED_VECTOR_T;" << std::endl;
    header << "typedef struct {unsigned int data[VECTOR_PACK_SIZE];}" << " PACKED_INDEX_T;" << std::endl;
    // End-of-row marker
    header << "#define IDX_MARKER 0xffffffff" << std::endl;
    // Semiring
    switch (this->semiring_) {
        case graphblas::kMulAdd:
            header << "#define MulAddSemiring" << std::endl;
            break;
        case graphblas::kLogicalAndOr:
            header << "#define LogicalAndOrSemiring" << std::endl;
            break;
        default:
            std::cerr << "Invalid semiring" << std::endl;
            break;
    }
    // Mask
    if (this->use_mask_) {
        header << "#define USE_MASK" << std::endl;
        switch (this->mask_type_) {
            case graphblas::kMaskWriteToZero:
                header << "#define MASK_WRITE_TO_ZERO" << std::endl;
                break;
            case graphblas::kMaskWriteToOne:
                header << "#define MASK_WRITE_TO_ONE" << std::endl;
                break;
            default:
                std::cerr << "Invalid mask type" << std::endl;
                break;
        }
    }
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
    size_t hbm_idx = 0;
    for (size_t i = 0; i < this->num_channels_; i++) {
        ini << "sp=kernel_spmv_1.channel_" << hbm_idx << "_partition_indptr:HBM[" << hbm_idx << "]" << std::endl;
        ini << "sp=kernel_spmv_1.channel_" << hbm_idx << "_indices:HBM[" << hbm_idx << "]" << std::endl;
        hbm_idx++;
        if (this->graph_is_weighed_) {
            ini << "sp=kernel_spmv_1.channel_" << hbm_idx << "_vals:HBM[" << hbm_idx << "]" << std::endl;
            hbm_idx++;
        }
    }
    // DDR
    ini << "sp=kernel_spmv_1.vector:DDR[0]" << std::endl;
    ini << "sp=kernel_spmv_1.out:DDR[0]" << std::endl;
    if (this->use_mask_) {
        ini << "sp=kernel_spmv_1.mask:DDR[0]" << std::endl;
    }
    ini.close();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::link_kernel_code() {
    std::string command;
    if (this->graph_is_weighed_) {
        command = "cp " + graphblas::root_path + "/graphblas/hw/kernel_spmv_weighted.cpp"
                + " " + graphblas::proj_folder_name + "/" + this->kernel_name_ + ".cpp";
    } else {
        command = "cp " + graphblas::root_path + "/graphblas/hw/kernel_spmv_unweighted.cpp"
                + " " + graphblas::proj_folder_name + "/" + this->kernel_name_ + ".cpp";
    }
    std::cout << command << std::endl;
    system(command.c_str());
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::load_and_format_matrix(CSRMatrix<float> const &csr_matrix_float) {
    this->csr_matrix_float_ = csr_matrix_float;
    this->csr_matrix_ = graphblas::io::csr_matrix_convert_from_float<matrix_data_t_internal_>(csr_matrix_float);
    SpMVDataFormatter<matrix_data_t_internal_, graphblas::pack_size, packed_data_t, graphblas::packed_index_t>
        formatter(this->csr_matrix_);
    matrix_data_t_internal_ val_marker = 0;
    formatter.format_pad_marker_end_of_row(this->out_buffer_len_,
                                           this->vector_buffer_len_,
                                           this->num_channels_,
                                           val_marker,
                                           graphblas::idx_marker);
    this->num_row_partitions_ = (this->csr_matrix_.num_rows + this->out_buffer_len_ - 1) /
        this->out_buffer_len_;
    this->num_col_partitions_ = (this->csr_matrix_.num_cols + this->vector_buffer_len_ - 1) /
        this->vector_buffer_len_;
    this->channel_indices_.resize(this->num_channels_);
    this->channel_partition_indptr_.resize(this->num_channels_);
    for (size_t c = 0; c < this->num_channels_; c++) {
        this->channel_partition_indptr_[c].resize(this->num_row_partitions_ * this->num_col_partitions_ + 1);
        this->channel_partition_indptr_[c][0] = 0;
    }
    if (this->graph_is_weighed_) {
        this->channel_vals_.resize(this->num_channels_);
    }
    for (size_t c = 0; c < this->num_channels_; c++) {
        for (size_t j = 0; j < this->num_row_partitions_; j++) {
            for (size_t i = 0; i < this->num_col_partitions_; i++) {
                auto indices_partition = formatter.get_packed_indices(j, i, c);
                this->channel_partition_indptr_[c][j*this->num_col_partitions_ + i + 1] =
                    this->channel_partition_indptr_[c][j*this->num_col_partitions_ + i]
                    + indices_partition.size();
                this->channel_indices_[c].insert(this->channel_indices_[c].end(),
                    indices_partition.begin(), indices_partition.end());
                if (this->graph_is_weighed_) {
                    auto vals_partition = formatter.get_packed_data(j, i, c);
                    this->channel_vals_[c].insert(this->channel_vals_[c].end(),
                        vals_partition.begin(), vals_partition.end());
                }
            }
        }
    }
    this->vector_.resize(this->csr_matrix_.num_cols);
    this->results_.resize(this->csr_matrix_.num_rows);
    std::fill(this->results_.begin(), this->results_.end(), 0);
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_matrix_host_to_device() {
    cl_int err;
    // Handle channel_partition_indptr and channel_indices
    cl_mem_ext_ptr_t channel_partition_indptr_ext[this->num_channels_];
    cl_mem_ext_ptr_t channel_indices_ext[this->num_channels_];
    this->channel_partition_indptr_buf.resize(this->num_channels_);
    this->channel_indices_buf.resize(this->num_channels_);
    for (size_t c = 0; c < this->num_channels_; c++) {
        channel_partition_indptr_ext[c].obj = this->channel_partition_indptr_[c].data();
        channel_partition_indptr_ext[c].param = 0;
        channel_partition_indptr_ext[c].flags = graphblas::HBM[c];
        channel_indices_ext[c].obj = this->channel_indices_[c].data();
        channel_indices_ext[c].param = 0;
        channel_indices_ext[c].flags = graphblas::HBM[c];
        OCL_CHECK(err, this->channel_partition_indptr_buf[c] = cl::Buffer(this->context_,
            CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
            sizeof(index_t) * this->channel_partition_indptr_[c].size(),
            &channel_partition_indptr_ext[c],
            &err));
        size_t channel_indices_size = sizeof(graphblas::packed_index_t) * this->channel_indices_[c].size();
        if (channel_indices_size >= 256 * 1000 * 1000) {
            std::cout << "The capcity of one HBM channel is 256 MB" << std::endl;
            exit(EXIT_FAILURE);
        }
        OCL_CHECK(err, this->channel_indices_buf[c] = cl::Buffer(this->context_,
            CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
            channel_indices_size,
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
            OCL_CHECK(err, this->channel_vals_buf[c] = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(packed_data_t) * this->channel_vals_[c].size(),
                &channel_vals_ext[c],
                &err));
        }
    }
    // Handle results
    cl_mem_ext_ptr_t results_ext;
    results_ext.obj = this->results_.data();
    results_ext.param = 0;
    results_ext.flags = graphblas::DDR[0];
    OCL_CHECK(err, this->results_buf = cl::Buffer(this->context_,
        CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(vector_data_t) * this->csr_matrix_.num_rows,
        &results_ext,
        &err));
    // Set arguments
    size_t arg_idx = 1; // the first argument (arg_idx = 0) is the dense vector
    for (size_t c = 0; c < this->num_channels_; c++) {
        OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->channel_partition_indptr_buf[c]));
        OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->channel_indices_buf[c]));
        if (this->graph_is_weighed_) {
            OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->channel_vals_buf[c]));
        }
    }
    if (this->use_mask_) {
        this->arg_idx_mask_ = arg_idx; // mask is right before results
        arg_idx++;
    }
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->results_buf));
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->csr_matrix_.num_rows));
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->csr_matrix_.num_cols));
    // Send data to device
    for (size_t c = 0; c < this->num_channels_; c++) {
        OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects(
            {this->channel_partition_indptr_buf[c]}, 0 /* 0 means from host to device */));
        OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects(
            {this->channel_indices_buf[c]}, 0));
        if (this->graph_is_weighed_) {
            OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects(
                {this->channel_vals_buf[c]}, 0));
        }
    }
    this->command_queue_.finish();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_vector_host_to_device(aligned_vector_t &vector) {
    this->vector_.assign(vector.begin(), vector.end());
    cl_mem_ext_ptr_t vector_ext;
    vector_ext.obj = this->vector_.data();
    vector_ext.param = 0;
    vector_ext.flags = graphblas::DDR[0];
    cl_int err;
    OCL_CHECK(err, this->vector_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * this->csr_matrix_.num_cols,
                &vector_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(0, this->vector_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->vector_buf}, 0));
    this->command_queue_.finish();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_mask_host_to_device(aligned_vector_t &mask) {
    this->mask_.assign(mask.begin(), mask.end());
    cl_mem_ext_ptr_t mask_ext;
    mask_ext.obj = this->mask_.data();
    mask_ext.param = 0;
    mask_ext.flags = graphblas::DDR[0];
    cl_int err;
    OCL_CHECK(err, this->mask_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * this->csr_matrix_.num_rows,
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
    index_t start = this->csr_matrix_float_.adj_indptr[row_idx];                  \
    index_t end = this->csr_matrix_float_.adj_indptr[row_idx + 1];                \
    for (size_t i = start; i < end; i++) {                                        \
        index_t index = this->csr_matrix_float_.adj_indices[i];                   \
        stmt;                                                                     \
    }                                                                             \
}                                                                               } \

template<typename matrix_data_t, typename vector_data_t>
graphblas::aligned_float_t
SpMVModule<matrix_data_t, vector_data_t>::compute_reference_results(aligned_float_t &vector) {
    aligned_float_t reference_results(this->csr_matrix_.num_rows);
    std::fill(reference_results.begin(), reference_results.end(), 0);
    switch (this->semiring_) {
        case graphblas::kMulAdd:
            if (this->graph_is_weighed_) {
                SPMV(reference_results[row_idx] += this->csr_matrix_float_.adj_data[i] * vector[index]);
            } else {
                SPMV(reference_results[row_idx] += vector[index]);
            }
            break;
        case graphblas::kLogicalAndOr:
            if (this->graph_is_weighed_) {
                SPMV(reference_results[row_idx] = reference_results[row_idx]
                     || (this->csr_matrix_float_.adj_data[i] && vector[index]));
            } else {
                SPMV(reference_results[row_idx] = reference_results[row_idx] || vector[index]);
            }
            break;
        default:
            std::cerr << "Invalid semiring" << std::endl;
            break;
    }
    return reference_results;
}


template<typename matrix_data_t, typename vector_data_t>
graphblas::aligned_float_t
SpMVModule<matrix_data_t, vector_data_t>::compute_reference_results(graphblas::aligned_float_t &vector,
                                                                    graphblas::aligned_float_t &mask) {
    graphblas::aligned_float_t reference_results = this->compute_reference_results(vector);
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


} // namespace module
} // namespace graphblas

#endif // __GRAPHBLAS_SPMV_MODULE_H
