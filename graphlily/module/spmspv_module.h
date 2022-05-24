#ifndef GRAPHLILY_SPMSPV_MODULE_H_
#define GRAPHLILY_SPMSPV_MODULE_H_

#include <cstdint>
#include <vector>
#include <fstream>
#include <chrono>

#include "xcl2.hpp"

#include "graphlily/global.h"
#include "graphlily/io/data_loader.h"
#include "graphlily/io/data_formatter.h"
#include "graphlily/module/base_module.h"


using graphlily::io::CSCMatrix;
using graphlily::io::FormattedCSCMatrix;
using graphlily::io::ColumnCyclicSplitCSC;
using graphlily::io::formatCSC;


namespace graphlily {
namespace module {

template<typename matrix_data_t, typename vector_data_t, typename idx_val_t>
class SpMSpVModule : public BaseModule {
private:
    /*! \brief The mask type */
    graphlily::MaskType mask_type_;
    /*! \brief The semiring */
    graphlily::SemiringType semiring_;
    /*! \brief The number of HBM channels */
    uint32_t spmspv_num_channels_;
    /*! \brief The length of output buffer of the kernel */
    uint32_t out_buf_len_;
    /*! \brief The number of row partitions */
    uint32_t num_row_partitions_;
    /*! \brief The number of columns for each channel partition */
    std::vector<uint32_t> num_cols_each_channel_;
    /*! \brief The index of the first argument after SpMSpV matrix, i.e. SpMSpV
     *         vector input is as 0 index.
     */
    uint32_t arg_index_offset_;

    using packet_t = struct {graphlily::idx_t indices[graphlily::pack_size];
                             matrix_data_t vals[graphlily::pack_size];};

    using aligned_idx_t = std::vector<graphlily::idx_t, aligned_allocator<graphlily::idx_t>>;
    using aligned_dense_vec_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;
    using aligned_sparse_vec_t = std::vector<idx_val_t, aligned_allocator<idx_val_t>>;
    using aligned_packet_t = std::vector<packet_t, aligned_allocator<packet_t>>;

    /*! \brief Matrix packets (indices + vals) */
    std::vector<aligned_packet_t> channel_packets_;
    /*! \brief Matrix indptr */
    std::vector<aligned_idx_t> channel_indptr_;
    /*! \brief Matrix partptr */
    std::vector<aligned_idx_t> channel_partptr_;
    /*! \brief Internal copy of the sparse vector.
               The index field of the first element is the non-zero count of the vector */
    aligned_sparse_vec_t vector_;
    /*! \brief Internal copy of mask */
    aligned_dense_vec_t mask_;
    /*! \brief The kernel results.
               The index field of the first element is the non-zero count of the results */
    aligned_sparse_vec_t results_;
    /*! \brief The nnz of the results */
    aligned_sparse_vec_t results_nnz_; // idx_val_t scalar. bypass XRT unaligned pointer warnings
    /*! \brief The sparse matrix using float data type */
    CSCMatrix<float> csc_matrix_float_;
    /*! \brief The sparse matrix */
    CSCMatrix<matrix_data_t> csc_matrix_;

public:
    // Device buffers
    std::vector<cl::Buffer> channel_packets_buf;
    std::vector<cl::Buffer> channel_indptr_buf;
    std::vector<cl::Buffer> channel_partptr_buf;
    cl::Buffer vector_buf;
    cl::Buffer mask_buf;
    cl::Buffer results_buf;
    cl::Buffer results_nnz_buf;

private:
    /*!
     * \brief The matrix data type should be the same as the vector data type.
     */
    void _check_data_type();

public:
    SpMSpVModule(uint32_t out_buf_len) : BaseModule() {
        this->_check_data_type();
        this->spmspv_num_channels_ = graphlily::spmspv_num_hbm_channels;
        this->arg_index_offset_ = 4 + 3 * this->spmspv_num_channels_;
        this->out_buf_len_ = out_buf_len;
    }

    /* SpMSpV apply overlay argument list: (k = 4 + 3 * HBM channels)
    * Index       Argument                              Used in this module?
    * 0           vector for spmv                       n
    * 1           mask for spmv (read port)             n
    * 2           mask for spmv (write port)            n
    * 3           output for spmv                       n
    *
    * 4~6         matrix for spmspv in HBM[23]          y
    * 7~9         matrix for spmspv in HBM[24]          y/n
    * ...         matrix for spmspv in HBM[??]          y/n
    *
    * k+0         vector for spmspv                     y
    * k+1         mask for spmspv                       y
    * k+2         output for spmspv                     y
    *
    * k+3         number of rows                        y
    * k+4         number of columns                     y
    * k+5         semiring operation type               y
    *
    * k+6         mask type                             y
    * k+7         overlay mode select                   y
    * k+8         apply vector length                   n
    * k+9         apply input value or semiring zero    y
    */
    void set_unused_args() override {
        for (size_t i = 0; i <= 3; ++i) {
            this->spmspv_apply_.setArg(i, cl::Buffer(this->context_, 0, 4));
        }
        if (this->mask_type_ == graphlily::kNoMask) {
            this->spmspv_apply_.setArg(arg_index_offset_ + 1, cl::Buffer(this->context_, 0, 4));
        }
        this->spmspv_apply_.setArg(arg_index_offset_ + 8, (unsigned)NULL);
    }

    void set_mode() override {
        this->spmspv_apply_.setArg(arg_index_offset_ + 7, 2);  // 2 is SpMSpV
    }

    /*!
     * \brief Get the number of rows of the sparse matrix.
     * \return The number of rows.
     */
    uint32_t get_num_rows() {
        return this->csc_matrix_.num_rows;
    }

    /*!
     * \brief Get the number of columns of the sparse matrix.
     * \return The number of columns.
     */
    uint32_t get_num_cols() {
        return this->csc_matrix_.num_cols;
    }

    /*!
     * \brief Get the number of non-zeros of the sparse matrix.
     * \return The number of non-zeros.
     */
    uint32_t get_nnz() {
        return this->csc_matrix_.adj_indptr[this->csc_matrix_.num_rows];
    }

    /*!
     * \brief Set the semiring type.
     * \param semiring The semiring type.
     */
    void set_semiring(graphlily::SemiringType semiring) {
        this->semiring_ = semiring;
    }

    /*!
     * \brief Set the mask type.
     * \param mask_type The mask type.
     */
    void set_mask_type(graphlily::MaskType mask_type) {
        this->mask_type_ = mask_type;
    }

    /*!
     * \brief Load a csc matrix and format the csr matrix.
     *        The csc matrix should have float data type.
     *        Data type conversion, if required, is handled internally.
     * \param csc_matrix_float The csc matrix using float data type.
     */
    void load_and_format_matrix(CSCMatrix<float> const &csc_matrix_float);

    /*!
     * \brief Send the formatted matrix from host to device.
     */
    void send_matrix_host_to_device();

    /*!
     * \brief Send the sparse vector from host to device.
     */
    void send_vector_host_to_device(aligned_sparse_vec_t &vector);

    /*!
     * \brief Send the mask from host to device.
     */
    void send_mask_host_to_device(aligned_dense_vec_t &mask);

    /*!
     * \brief Run the module.
     */
    void run();

    /*!
     * \brief Send the sparse vector from device to host.
     */
    aligned_sparse_vec_t send_vector_device_to_host() {
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
    aligned_sparse_vec_t send_results_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->results_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->results_;
    }

    /*!
     * \brief Get the nnz of the results. The host uses the information to do the scheduling.
     */
    uint32_t get_results_nnz() {
        this->copy_buffer_device_to_device(this->results_buf, this->results_nnz_buf, sizeof(idx_val_t) * 1);
        return this->results_nnz_[0].index;
    }

    /*!
     * \brief Compute reference results.
     * \param vector The sparse vector.
     * \param mask The dense mask.
     * \return The reference results in dense format.
     */
    graphlily::aligned_dense_float_vec_t compute_reference_results(
        graphlily::aligned_sparse_float_vec_t &vector,
        graphlily::aligned_dense_float_vec_t &mask
    );
};


template<typename matrix_data_t, typename vector_data_t, typename idx_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, idx_val_t>::_check_data_type() {
    assert((std::is_same<matrix_data_t, vector_data_t>::value));
}


template<typename matrix_data_t, typename vector_data_t, typename idx_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, idx_val_t>::load_and_format_matrix(
        CSCMatrix<float> const &csc_matrix_float) {
    this->csc_matrix_float_ = csc_matrix_float;
    this->csc_matrix_ = graphlily::io::csc_matrix_convert_from_float<matrix_data_t>(csc_matrix_float);

    std::vector<CSCMatrix<matrix_data_t> > csc_matrices = ColumnCyclicSplitCSC<matrix_data_t>(this->csc_matrix_,
                                                                                              this->spmspv_num_channels_);
    FormattedCSCMatrix<packet_t> formatted_csc_matrices[this->spmspv_num_channels_];
    this->channel_packets_.resize(this->spmspv_num_channels_);
    this->channel_indptr_.resize(this->spmspv_num_channels_);
    this->channel_partptr_.resize(this->spmspv_num_channels_);
    this->num_cols_each_channel_.resize(this->spmspv_num_channels_);
    for (uint32_t c = 0; c < this->spmspv_num_channels_; c++) {
        formatted_csc_matrices[c] = formatCSC<matrix_data_t, packet_t>(csc_matrices[c],
                                                                       this->semiring_,
                                                                       graphlily::pack_size,
                                                                       this->out_buf_len_);
        this->channel_packets_[c] = formatted_csc_matrices[c].get_formatted_packet();
        this->channel_indptr_[c] = formatted_csc_matrices[c].get_formatted_indptr();
        this->channel_partptr_[c] = formatted_csc_matrices[c].get_formatted_partptr();
        this->num_cols_each_channel_[c] = formatted_csc_matrices[c].num_cols;
    }
    this->num_row_partitions_ = formatted_csc_matrices[0].num_row_partitions;

    // reserve enough space for the vector
    this->vector_.resize(this->get_num_cols() + 1);

    // prepare output memory
    this->results_.resize(this->get_num_rows() + 1);
    this->results_nnz_.resize(1);

    std::fill(this->results_.begin(), this->results_.end(), (idx_val_t){0, 0});
}


template<typename matrix_data_t, typename vector_data_t, typename idx_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, idx_val_t>::send_matrix_host_to_device() {
    cl_int err;
    // Handle matrix packet, indptr and partptr
    cl_mem_ext_ptr_t channel_packets_ext[this->spmspv_num_channels_];
    cl_mem_ext_ptr_t channel_indptr_ext[this->spmspv_num_channels_];
    cl_mem_ext_ptr_t channel_partptr_ext[this->spmspv_num_channels_];

    this->channel_packets_buf.resize(this->spmspv_num_channels_);
    this->channel_indptr_buf.resize(this->spmspv_num_channels_);
    this->channel_partptr_buf.resize(this->spmspv_num_channels_);

    for (size_t c = 0; c < this->spmspv_num_channels_; c++) {
        channel_packets_ext[c].obj = this->channel_packets_[c].data();
        channel_packets_ext[c].param = 0;
        channel_packets_ext[c].flags = graphlily::HBM[c + 23];

        channel_indptr_ext[c].obj = this->channel_indptr_[c].data();
        channel_indptr_ext[c].param = 0;
        channel_indptr_ext[c].flags = graphlily::HBM[c + 23];

        channel_partptr_ext[c].obj = this->channel_partptr_[c].data();
        channel_partptr_ext[c].param = 0;
        channel_partptr_ext[c].flags = graphlily::HBM[c + 23];

        size_t channel_packets_size = sizeof(packet_t) * this->channel_packets_[c].size()
                                      + sizeof(unsigned) * this->channel_indptr_[c].size()
                                      + sizeof(unsigned) * this->channel_partptr_[c].size();
        std::cout << "channel_packets_size: " << channel_packets_size << std::endl;
        if (channel_packets_size >= 256 * 1024 * 1024) {
            std::cout << "The capcity of one HBM channel is 256 MB" << std::endl;
            exit(EXIT_FAILURE);
        }

        OCL_CHECK(err, this->channel_packets_buf[c] = cl::Buffer(this->context_,
            CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
            sizeof(packet_t) * this->channel_packets_[c].size(),
            &channel_packets_ext[c],
            &err));
        OCL_CHECK(err, this->channel_indptr_buf[c] = cl::Buffer(this->context_,
            CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
            sizeof(graphlily::idx_t) * (this->num_cols_each_channel_[c] + 1) * this->num_row_partitions_ ,
            &channel_indptr_ext[c],
            &err));
        OCL_CHECK(err, this->channel_partptr_buf[c] = cl::Buffer(this->context_,
            CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
            sizeof(graphlily::idx_t) * (this->num_row_partitions_ + 1),
            &channel_partptr_ext[c],
            &err));
    }

    for (size_t c = 0; c < this->spmspv_num_channels_; c++) {
        OCL_CHECK(err, err = this->spmspv_apply_.setArg(4 + 3*c, this->channel_packets_buf[c]));
        OCL_CHECK(err, err = this->spmspv_apply_.setArg(5 + 3*c, this->channel_indptr_buf[c]));
        OCL_CHECK(err, err = this->spmspv_apply_.setArg(6 + 3*c, this->channel_partptr_buf[c]));
    }

    OCL_CHECK(err, err = this->spmspv_apply_.setArg(arg_index_offset_ + 3, this->csc_matrix_.num_rows));
    OCL_CHECK(err, err = this->spmspv_apply_.setArg(arg_index_offset_ + 4, this->csc_matrix_.num_cols));
    OCL_CHECK(err, err = this->spmspv_apply_.setArg(arg_index_offset_ + 5, (char)this->semiring_.op));
    OCL_CHECK(err, err = this->spmspv_apply_.setArg(arg_index_offset_ + 6, (char)this->mask_type_));

    unsigned zero = graphlily::pack_raw_bits_to_uint(this->semiring_.zero);
    OCL_CHECK(err, err = this->spmspv_apply_.setArg(arg_index_offset_ + 9, zero));

    for (size_t c = 0; c < this->spmspv_num_channels_; c++) {
        OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({
            this->channel_packets_buf[c],
            this->channel_indptr_buf[c],
            this->channel_partptr_buf[c],
            }, 0 /* 0 means from host*/));
    }

    this->command_queue_.finish();
    // std::cout << "INFO: [Module SpMSpV - send matrix] matrix successfully send to device."
    //           << std::endl << std::flush;

    // Handle results
    cl_mem_ext_ptr_t results_ext;
    results_ext.obj = this->results_.data();
    results_ext.param = 0;
    results_ext.flags = graphlily::HBM[22];

    OCL_CHECK(err, this->results_buf = cl::Buffer(this->context_,
        CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(idx_val_t) * (this->get_num_rows() + 1),
        &results_ext,
        &err));

    cl_mem_ext_ptr_t results_nnz_ext;
    results_nnz_ext.obj = this->results_nnz_.data();
    results_nnz_ext.param = 0;
    results_nnz_ext.flags = graphlily::HBM[22];

    OCL_CHECK(err, this->results_nnz_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(idx_val_t) * 1,
                &results_nnz_ext,
                &err));

    OCL_CHECK(err, err = this->spmspv_apply_.setArg(arg_index_offset_ + 2, this->results_buf));
    // std::cout << "INFO: [Module SpMSpV - allocate result] space for result successfully allocated on device."
    //           << std::endl << std::flush;
}


template<typename matrix_data_t, typename vector_data_t, typename idx_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, idx_val_t>::send_vector_host_to_device(
        aligned_sparse_vec_t &vector) {
    cl_int err;

    // copy the input vector
    std::copy(vector.begin(), vector.end(), this->vector_.begin());

    // Handle vector
    cl_mem_ext_ptr_t vector_ext;
    vector_ext.obj = this->vector_.data();
    vector_ext.param = 0;
    vector_ext.flags = graphlily::HBM[20];

    OCL_CHECK(err, this->vector_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(idx_val_t) * this->vector_.size(),
                &vector_ext,
                &err));

    // set argument
    OCL_CHECK(err, err = this->spmspv_apply_.setArg(arg_index_offset_ + 0, this->vector_buf));

    // Send vector to device
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->vector_buf}, 0));
    this->command_queue_.finish();
    // std::cout << "INFO: [Module SpMSpV - send vector] vector successfully send to device."
    //           << std::endl << std::flush;
}


template<typename matrix_data_t, typename vector_data_t, typename idx_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, idx_val_t>::send_mask_host_to_device(
        aligned_dense_vec_t &mask) {
    cl_int err;

    // copy the input mask
    this->mask_.resize(mask.size());
    std::copy(mask.begin(), mask.end(), this->mask_.begin());

    // Handle mask
    cl_mem_ext_ptr_t mask_ext;
    mask_ext.obj = this->mask_.data();
    mask_ext.param = 0;
    mask_ext.flags = graphlily::HBM[21];

    OCL_CHECK(err, this->mask_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * this->mask_.size(),
                &mask_ext,
                &err));

    // set argument
    OCL_CHECK(err, err = this->spmspv_apply_.setArg(arg_index_offset_ + 1, this->mask_buf));

    // Send mask to device
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->mask_buf}, 0));
    this->command_queue_.finish();
    // std::cout << "INFO: [Module SpMSpV - send mask] mask successfully send to device."
    //           << std::endl << std::flush;
}


template<typename matrix_data_t, typename vector_data_t, typename idx_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, idx_val_t>::run() {
    cl_int err;
    OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->spmspv_apply_));
    this->command_queue_.finish();
}


template<typename matrix_data_t, typename vector_data_t, typename idx_val_t>
graphlily::aligned_dense_float_vec_t
SpMSpVModule<matrix_data_t, vector_data_t, idx_val_t>::compute_reference_results(
        graphlily::aligned_sparse_float_vec_t &vector,
        graphlily::aligned_dense_float_vec_t &mask) {
    // define inf based on val_t
    float inf;
    if (std::is_same<graphlily::val_t, float>::value) {
        inf = float(graphlily::FLOAT_INF);
    } else if (std::is_same<graphlily::val_t, unsigned>::value) {
        inf = float(graphlily::UINT_INF);
    } else {
        inf = float(graphlily::UFIXED_INF);
    }

    // measure dimensions
    unsigned vec_nnz_total = vector[0].index;
    unsigned num_rows = this->csc_matrix_.num_rows;

    // create result container
    aligned_dense_float_vec_t reference_results(num_rows, this->semiring_.zero);

    // indices of active columns are stored in vec_idx
    // number of active columns = vec_nnz_total
    // loop over all active columns
    for (unsigned active_colid = 0; active_colid < vec_nnz_total; active_colid++) {

        float nnz_from_vec = vector[active_colid + 1].val;
        idx_t current_colid = vector[active_colid + 1].index;

        // slice out the current column out of the active columns
        idx_t col_start = this->csc_matrix_.adj_indptr[current_colid];
        idx_t col_end = this->csc_matrix_.adj_indptr[current_colid + 1];

        // loop over all nnzs in the current column
        for (unsigned mat_element_id = col_start; mat_element_id < col_end; mat_element_id++) {
            idx_t current_row_id = this->csc_matrix_.adj_indices[mat_element_id];
            float nnz_from_mat = this->csc_matrix_.adj_data[mat_element_id];
            float incr;
            switch (this->semiring_.op) {
                case kMulAdd:
                    incr = nnz_from_mat * nnz_from_vec;
                    reference_results[current_row_id] += incr;
                    break;
                case kLogicalAndOr:
                    incr = (nnz_from_mat && nnz_from_vec);
                    reference_results[current_row_id] = reference_results[current_row_id] || incr;
                    break;
                case kAddMin:
                    // simulate the AP_SAT oveflow mode
                    if (nnz_from_mat > inf || nnz_from_vec > inf) {
                        incr = inf;
                    } else {
                        incr = std::min(nnz_from_mat + nnz_from_vec, inf);
                    }
                    reference_results[current_row_id] = std::min(reference_results[current_row_id], incr);
                    break;
                default:
                    std::cerr << "ERROR: [Module SpMSpV] Invalid semiring" << std::endl;
                    break;
            }
        }
    }
    // mask off values
    for (unsigned i = 0; i < num_rows; i++) {
        bool mask_off;
        switch (this->mask_type_) {
            case kNoMask:
                mask_off = false;
                break;
            case kMaskWriteToOne:
                mask_off = (mask[i] == 0);
                break;
            case kMaskWriteToZero:
                mask_off = (mask[i] != 0);
                break;
            default:
                mask_off = true;
                break;
        }
        if (mask_off) reference_results[i] = this->semiring_.zero;
    }
    // std::cout << "INFO: [Module SpMSpV - compute reference] reference computation successfully complete."
    //           << std::endl << std::flush;
    return reference_results;
}

}  // namespace module
}  // namespace graphlily

#endif  // GRAPHLILY_SPMSPV_MODULE_H_
