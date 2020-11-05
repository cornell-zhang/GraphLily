#ifndef __GRAPHBLAS_SPMSPV_MODULE_H
#define __GRAPHBLAS_SPMSPV_MODULE_H

#include <cstdint>
#include <vector>
#include <fstream>
#include <chrono>

#include "xcl2.hpp"

#include "../global.h"
#include "../io/data_loader.h"
#include "../io/data_formatter.h"
#include "./base_module.h"

using graphblas::io::CSCMatrix;
using graphblas::io::FormattedCSCMatrix;
using graphblas::io::formatCSC;


namespace graphblas {
namespace module {

unsigned log2(unsigned x) {
    switch (x) {
        case    1: return 0;
        case    2: return 1;
        case    4: return 2;
        case    8: return 3;
        case   16: return 4;
        default  : return 0;
    }
}

template<typename matrix_data_t, typename vector_data_t, typename index_val_t>
class SpMSpVModule : public BaseModule {
private:
    /*! \brief The mask type */
    graphblas::MaskType mask_type_;
    /*! \brief The semiring */
    graphblas::SemiringType semiring_;
    /*! \brief The length of output buffer of the kernel */
    uint32_t out_buf_len_;
    /*! \brief The number of row partitions */
    uint32_t num_row_partitions_;
    /*! \brief The number of packets */
    uint32_t num_packets_;

    // using val_t = vector_data_t;
    // using index_val_t = struct {val_t val; idx_t index;};
    using packet_t = struct {graphblas::idx_t indices[graphblas::pack_size]; matrix_data_t vals[graphblas::pack_size];};

    using aligned_idx_t = std::vector<graphblas::idx_t, aligned_allocator<graphblas::idx_t>>;
    using aligned_dense_vec_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;
    using aligned_sparse_vec_t = std::vector<index_val_t, aligned_allocator<index_val_t>>;
    using aligned_packet_t = std::vector<packet_t, aligned_allocator<packet_t>>;

    /*! \brief Matrix packets (indices + vals) */
    aligned_packet_t channel_packets_;
    /*! \brief Matrix indptr */
    aligned_idx_t channel_indptr_;
    /*! \brief Matrix partptr */
    aligned_idx_t channel_partptr_;
    /*! \brief Internal copy of the sparse vector.
               The index field of the first element is the non-zero count of the vector */
    aligned_sparse_vec_t vector_;
    /*! \brief Internal copy of mask */
    aligned_dense_vec_t mask_;
    /*! \brief The argument index of mask to be used in setArg */
    uint32_t arg_idx_mask_;
    /*! \brief The kernel results.
               The index field of the first element is the non-zero count of the results */
    aligned_sparse_vec_t results_;
    /*! \brief The sparse matrix using float data type*/
    CSCMatrix<float> csc_matrix_float_;
    /*! \brief The sparse matrix */
    CSCMatrix<matrix_data_t> csc_matrix_;
    /*! \brief The formatted matrix */
    FormattedCSCMatrix<packet_t> formatted_csc_matrix_;

public:
    // Device buffers
    cl::Buffer channel_packets_buf;
    cl::Buffer channel_indptr_buf;
    cl::Buffer channel_partptr_buf;
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
     * \param semiring The semiring.
     * \param out_buf_len The length of output buffer.
     */
    void _get_kernel_config(graphblas::SemiringType semiring,
                            uint32_t out_buf_len);

public:
    SpMSpVModule(uint32_t out_buf_len) : BaseModule("kernel_spmspv") {
        this->_check_data_type();
        this->out_buf_len_ = out_buf_len;
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
     * \brief Change the kernel semiring and masktype
     */
    void config_kernel(graphblas::SemiringType semiring, graphblas::MaskType masktype) {
        cl_int err;
        this->semiring_ = semiring;
        this->mask_type_ = masktype;
        // Set argument
        OCL_CHECK(err, err = this->kernel_.setArg(8, (char)this->semiring_.op));
        OCL_CHECK(err, err = this->kernel_.setArg(9, (char)this->mask_type_));
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
        // truncate useless data
        this->vector_.resize(this->vector_[0].index + 1);
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
        // truncate useless data
        this->results_.resize(this->results_[0].index + 1);
        std::cout << "INFO: [Module SpMSpV - collect result] result collected." << std::endl << std::flush;
        return this->results_;
    }

    /*!
     * \brief Compute reference results.
     * \param vector The sparse vector.
     * \param mask The dense mask.
     * \return The reference results in dense format.
     */
    graphblas::aligned_dense_float_vec_t compute_reference_results(graphblas::aligned_sparse_float_vec_t &vector,
                                                                   graphblas::aligned_dense_float_vec_t &mask);

    void generate_kernel_header() override;

    void generate_kernel_ini() override;
};


template<typename matrix_data_t, typename vector_data_t, typename index_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, index_val_t>::_check_data_type() {
    assert((std::is_same<matrix_data_t, vector_data_t>::value));
}


template<typename matrix_data_t, typename vector_data_t, typename index_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, index_val_t>::_get_kernel_config(graphblas::SemiringType semiring,
                                                                                 uint32_t out_buf_len) {
    this->semiring_ = semiring;
    this->out_buf_len_ = out_buf_len;
}


template<typename matrix_data_t, typename vector_data_t, typename index_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, index_val_t>::generate_kernel_header() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream header(graphblas::proj_folder_name + "/" + this->kernel_name_ + ".h", std::ios_base::app);
    // Kernel configuration
    header << "const unsigned OUT_BUF_LEN = " << this->out_buf_len_ << ";" << std::endl;
    header.close();
}


template<typename matrix_data_t, typename vector_data_t, typename index_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, index_val_t>::generate_kernel_ini() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream ini(graphblas::proj_folder_name + "/" + this->kernel_name_ + ".ini");

    // memory channel connectivity
    ini << "[connectivity]" << std::endl;

    // allocate matrix on DDR1
    ini << "sp=kernel_spmspv_1.matrix:DDR[1]" << std::endl;
    ini << "sp=kernel_spmspv_1.mat_indptr:DDR[1]" << std::endl;
    ini << "sp=kernel_spmspv_1.mat_partptr:DDR[1]" << std::endl;

    // allocate others on DDR0
    ini << "sp=kernel_spmspv_1.mask:DDR[0]" << std::endl;
    ini << "sp=kernel_spmspv_1.vector:DDR[0]" << std::endl;
    ini << "sp=kernel_spmspv_1.result:DDR[0]" << std::endl;

    // enable retiming
    ini << "[vivado]" << std::endl;
    ini << "prop=run.__KERNEL__.{STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS}={-retiming}" << std::endl;

    ini.close();
}


template<typename matrix_data_t, typename vector_data_t, typename index_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, index_val_t>::load_and_format_matrix(CSCMatrix<float> const &csc_matrix_float) {
    this->csc_matrix_float_ = csc_matrix_float;
    this->csc_matrix_ = graphblas::io::csc_matrix_convert_from_float<matrix_data_t>(csc_matrix_float);
    this->formatted_csc_matrix_ = formatCSC<matrix_data_t, packet_t>(this->csc_matrix_, this->semiring_, graphblas::pack_size, this->out_buf_len_);
    this->num_row_partitions_ = this->formatted_csc_matrix_.num_row_partitions;
    this->num_packets_ = this->formatted_csc_matrix_.num_packets_total;

    this->channel_packets_ = this->formatted_csc_matrix_.get_formatted_packet();
    this->channel_indptr_ = this->formatted_csc_matrix_.get_formatted_indptr();
    this->channel_partptr_ = this->formatted_csc_matrix_.get_formatted_partptr();

    // reserve enough space for the vector
    this->vector_.resize(this->get_num_cols() + 1);

    // prepare output memory
    this->results_.resize(this->get_num_rows() + 1);

    std::fill(this->results_.begin(), this->results_.end(), (index_val_t){0,0});
}


template<typename matrix_data_t, typename vector_data_t, typename index_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, index_val_t>::send_matrix_host_to_device() {
    cl_int err;
    // Handle matrix packet, indptr and partptr
    cl_mem_ext_ptr_t channel_packets_ext;
    cl_mem_ext_ptr_t channel_indptr_ext;
    cl_mem_ext_ptr_t channel_partptr_ext;

    channel_packets_ext.obj = this->channel_packets_.data();
    channel_packets_ext.param = 0;
    channel_packets_ext.flags = graphblas::DDR[1];

    channel_indptr_ext.obj = this->channel_indptr_.data();
    channel_indptr_ext.param = 0;
    channel_indptr_ext.flags = graphblas::DDR[1];

    channel_partptr_ext.obj = this->channel_partptr_.data();
    channel_partptr_ext.param = 0;
    channel_partptr_ext.flags = graphblas::DDR[1];

    // Allocate memory on the FPGA
    OCL_CHECK(err, this->channel_packets_buf = cl::Buffer(this->context_,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(packet_t) * this->num_packets_,
        &channel_packets_ext,
        &err));
    OCL_CHECK(err, this->channel_indptr_buf = cl::Buffer(this->context_,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphblas::idx_t) * (this->get_num_cols() + 1) * this->num_row_partitions_ ,
        &channel_indptr_ext,
        &err));
    OCL_CHECK(err, this->channel_partptr_buf = cl::Buffer(this->context_,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphblas::idx_t) * (this->num_row_partitions_ + 1),
        &channel_partptr_ext,
        &err));

    // Set arguments
    OCL_CHECK(err, err = this->kernel_.setArg(0, this->channel_packets_buf));
    OCL_CHECK(err, err = this->kernel_.setArg(1, this->channel_indptr_buf));
    OCL_CHECK(err, err = this->kernel_.setArg(2, this->channel_partptr_buf));
    OCL_CHECK(err, err = this->kernel_.setArg(6, this->csc_matrix_.num_rows));
    OCL_CHECK(err, err = this->kernel_.setArg(7, this->csc_matrix_.num_cols));

    // Send data to device
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({
                                this->channel_packets_buf,
                                this->channel_indptr_buf,
                                this->channel_partptr_buf,
                                }, 0 /* 0 means from host*/)
    );
    this->command_queue_.finish();
    std::cout << "INFO: [Module SpMSpV - send matrix] matrix successfully send to device." << std::endl << std::flush;

    // Handle results
    cl_mem_ext_ptr_t results_ext;

    results_ext.obj = this->results_.data();
    results_ext.param = 0;
    results_ext.flags = graphblas::DDR[0];

    // Allocate memory on the FPGA
    OCL_CHECK(err, this->results_buf = cl::Buffer(this->context_,
        CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(index_val_t) * (this->get_num_rows() + 1),
        &results_ext,
        &err));

    // Set argument
    OCL_CHECK(err, err = this->kernel_.setArg(5, this->results_buf));
    std::cout << "INFO: [Module SpMSpV - allocate result] space for result successfully allocated on device." << std::endl << std::flush;
}


template<typename matrix_data_t, typename vector_data_t, typename index_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, index_val_t>::send_vector_host_to_device(aligned_sparse_vec_t &vector) {
    cl_int err;

    // copy the input vector
    this->vector_.resize(vector.size());
    std::copy(vector.begin(), vector.end(), this->vector_.begin());
    // std::cout << "INFO: [Module SpMSpV - send vector] vector successfully loaded to host memory." << std::endl << std::flush;
    // std::cout << "  vector size : " << vector.size() << std::endl << std::flush;
    // std::cout << "  vector Nnz : "  << vector[0].index << std::endl << std::flush;

    // Handle vector
    cl_mem_ext_ptr_t vector_ext;
    vector_ext.obj = this->vector_.data();
    vector_ext.param = 0;
    vector_ext.flags = graphblas::DDR[0];

    OCL_CHECK(err, this->vector_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(index_val_t) * this->vector_.size(),
                &vector_ext,
                &err));
    // std::cout << "INFO: [Module SpMSpV - send vector] memory allocated on FPGA" << std::endl << std::flush;
    // set argument
    OCL_CHECK(err, err = this->kernel_.setArg(3, this->vector_buf));

    // Send data to device
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->vector_buf}, 0));
    this->command_queue_.finish();
    std::cout << "INFO: [Module SpMSpV - send vector] vector successfully send to device." << std::endl << std::flush;
}


template<typename matrix_data_t, typename vector_data_t, typename index_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, index_val_t>::send_mask_host_to_device(aligned_dense_vec_t &mask) {
    cl_int err;

    // copy the input vector
    this->mask_.resize(mask.size());
    std::copy(mask.begin(), mask.end(), this->mask_.begin());
    // std::cout << "INFO: [Module SpMSpV - send mask] mask successfully loaded to host memory." << std::endl << std::flush;

    // Handle vector
    cl_mem_ext_ptr_t vector_ext;
    vector_ext.obj = this->mask_.data();
    vector_ext.param = 0;
    vector_ext.flags = graphblas::DDR[0];

    OCL_CHECK(err, this->mask_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * this->mask_.size(),
                &vector_ext,
                &err));
    // std::cout << "INFO: [Module SpMSpV - send mask] memory allocated on FPGA" << std::endl << std::flush;
    // set argument
    OCL_CHECK(err, err = this->kernel_.setArg(4, this->mask_buf));

    // Send data to device
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->mask_buf}, 0));
    this->command_queue_.finish();
    std::cout << "INFO: [Module SpMSpV - send vector] vector successfully send to device." << std::endl << std::flush;
}

template<typename matrix_data_t, typename vector_data_t, typename index_val_t>
void SpMSpVModule<matrix_data_t, vector_data_t, index_val_t>::run() {
    cl_int err;
    OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->kernel_));
    this->command_queue_.finish();
}

/*!
 * \brief Compute reference results.
 * \param vector The sparse vector.
 * \param mask The dense mask.
 * \return The reference results in dense format.
 */
template<typename matrix_data_t, typename vector_data_t, typename index_val_t>
graphblas::aligned_dense_float_vec_t
SpMSpVModule<matrix_data_t, vector_data_t, index_val_t>::compute_reference_results(graphblas::aligned_sparse_float_vec_t &vector,
                                                                                   graphblas::aligned_dense_float_vec_t &mask) {
    // measure dimensions
    unsigned vec_nnz_total = vector[0].index;
    unsigned num_rows = this->csc_matrix_.num_rows;

    // create result container
    aligned_dense_float_vec_t reference_results(num_rows,this->semiring_.zero);

    // indices of active columns are stored in vec_idx
    // number of active columns = vec_nnz_total
    // loop over all active columns
    for (unsigned active_colid = 0; active_colid < vec_nnz_total; active_colid++) {
        idx_t current_colid = vector[active_colid + 1].index;
        // slice out the current column out of the active columns
        idx_t col_id_start = this->csc_matrix_.adj_indptr[current_colid];
        idx_t col_id_end = this->csc_matrix_.adj_indptr[current_colid + 1];

        // loop over all nnzs in the current column
        for (unsigned mat_element_id = col_id_start; mat_element_id < col_id_end; mat_element_id++) {
            idx_t current_row_id = this->csc_matrix_.adj_indices[mat_element_id];
            float nnz_from_mat = this->csc_matrix_.adj_data[mat_element_id];
            float nnz_from_vec = vector[active_colid + 1].val;
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
                    incr = nnz_from_mat + nnz_from_vec;
                    reference_results[current_row_id] = (reference_results[current_row_id] < incr) ? reference_results[current_row_id] : incr;
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
                mask_off = (mask[i] == this->semiring_.zero);
                break;
            case kMaskWriteToZero:
                mask_off = (mask[i] != this->semiring_.zero);
                break;
            default:
                mask_off = true;
                break;
        }
        if (mask_off) reference_results[i] = this->semiring_.zero;
    }
    std::cout << "INFO: [Module SpMSpV - compute reference] reference computation successfully complete." << std::endl << std::flush;
    return reference_results;
}


} // namespace module
} // namespace graphblas

#endif // __GRAPHBLAS_SPMSPV_MODULE_H
