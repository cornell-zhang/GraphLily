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
using graphblas::io::SpMSpVDataFormatter;


namespace graphblas {
namespace module {

unsigned int log2(unsigned int x) {
    switch (x) {
        case    1: return 0;
        case    2: return 1;
        case    4: return 2;
        case    8: return 3;
        case   16: return 4;
        default  : return 0;
    }
}

template<typename matrix_data_t, typename vector_data_t, typename val_index_t>
class SpMSpVModule : public BaseModule {
private:
    /*! \brief Whether the kernel uses mask */
    bool use_mask_;
    /*! \brief The mask type */
    graphblas::MaskType mask_type_;
    /*! \brief The semiring */
    SemiRingType semiring_;
    /*! \brief The length of output buffer of the kernel */
    uint32_t out_buffer_len_;
    /*! \brief The number of row partitions */
    uint32_t num_row_partitions_;
    /*! \brief The number of packets */
    uint32_t num_packets_;

    // using val_t = vector_data_t;
    // using val_index_t = struct {val_t val; index_t index;};
    using packet_t = struct {graphblas::index_t indices[graphblas::pack_size]; matrix_data_t vals[graphblas::pack_size];};

    using aligned_index_t = std::vector<graphblas::index_t, aligned_allocator<graphblas::index_t>>;
    using aligned_val_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;
    using aligned_val_index_t = std::vector<val_index_t, aligned_allocator<val_index_t>>;
    using aligned_packet_t = std::vector<packet_t, aligned_allocator<packet_t>>;

    // String representation of the data type
    std::string val_t_str_;

    /*! \brief Matrix packets (indices + vals) */
    aligned_packet_t channel_packets_;
    /*! \brief Matrix indptr */
    aligned_index_t channel_indptr_;
    /*! \brief Matrix partptr */
    aligned_index_t channel_partptr_;
    /*! \brief Internal copy of the sparse vector.
               The index field of the first element is the non-zero count of the vector */
    aligned_val_index_t vector_;
    /*! \brief Internal copy of mask */
    aligned_val_t mask_;
    /*! \brief The argument index of mask to be used in setArg */
    uint32_t arg_idx_mask_;
    /*! \brief The kernel results.
               The index field of the first element is the non-zero count of the results */
    aligned_val_index_t results_;
    /*! \brief The sparse matrix using float data type*/
    CSCMatrix<float> csc_matrix_float_;
    /*! \brief The sparse matrix */
    CSCMatrix<matrix_data_t> csc_matrix_;

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
     * \param out_buffer_len The length of output buffer.
     */
    void _get_kernel_config(SemiRingType semiring,
                            uint32_t out_buffer_len);

public:
    SpMSpVModule(SemiRingType semiring,
                 uint32_t out_buffer_len) : BaseModule("kernel_spmspv") {
        this->_check_data_type();
        this->_get_kernel_config(semiring, out_buffer_len);
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
    void send_vector_host_to_device(aligned_val_index_t &vector);

    /*!
     * \brief Send the mask from host to device.
     */
    void send_mask_host_to_device(aligned_val_t &mask);

    /*!
     * \brief Run the module.
     */
    void run();

    /*!
     * \brief Send the sparse vector from device to host.
     */
    aligned_val_index_t send_vector_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->vector_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->vector_;
    }

    /*!
     * \brief Send the mask from device to host.
     * \return The mask.
     */
    aligned_val_t send_mask_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->mask_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->mask_;
    }

    /*!
     * \brief Send the results from device to host.
     * \return The results.
     */
    aligned_val_index_t send_results_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->results_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        // truncate useless data
        this->results_.resize(this->results_[0].index + 1);
        return this->results_;
    }

    /*!
     * \brief Compute reference results.
     * \param vector The sparse vector.
     * \return The reference results.
     */
    graphblas::aligned_sparse_float_vec_t compute_reference_results(graphblas::aligned_sparse_float_vec_t &vector);

    /*!
     * \brief Compute reference results.
     * \param vector The sparse vector.
     * \param mask The dense mask.
     * \return The reference results.
     */
    graphblas::aligned_sparse_float_vec_t compute_reference_results(graphblas::aligned_sparse_float_vec_t &vector,
                                                                    graphblas::aligned_dense_float_vec_t &mask);

    void generate_kernel_header() override;

    void generate_kernel_ini() override;
};


template<typename matrix_data_t, typename vector_data_t, typename val_index_t>
void SpMSpVModule<matrix_data_t, vector_data_t, val_index_t>::_check_data_type() {
    assert((std::is_same<matrix_data_t, vector_data_t>::value));
    this->val_t_str_ = graphblas::dtype_to_str<vector_data_t>();
}


template<typename matrix_data_t, typename vector_data_t, typename val_index_t>
void SpMSpVModule<matrix_data_t, vector_data_t, val_index_t>::_get_kernel_config(SemiRingType semiring,
                                                                                 uint32_t out_buffer_len) {
    this->semiring_ = semiring;
    this->out_buffer_len_ = out_buffer_len;
}


template<typename matrix_data_t, typename vector_data_t, typename val_index_t>
void SpMSpVModule<matrix_data_t, vector_data_t, val_index_t>::generate_kernel_header() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream header(graphblas::proj_folder_name + "/" + this->kernel_name_ + ".h");
    // Kernel configuration
    header << "const unsigned int PACKET_SIZE = " << graphblas::pack_size << ";" << std::endl;
    header << "const unsigned int NUM_PE = " << graphblas::pack_size << ";" << std::endl;
    header << "const unsigned int NUM_PORT_PER_BANK = 2;" << std::endl;
    header << "const unsigned int NUM_BANK = NUM_PE" << ";" << std::endl;
    header << "const unsigned int NUM_LANE = NUM_BANK * NUM_PORT_PER_BANK" << ";" << std::endl;
    header << "const unsigned int PE_ID_NBITS = "   << log2(graphblas::pack_size) << ";" << std::endl;
    header << "const unsigned int BANK_ID_NBITS = " << log2(graphblas::pack_size) << ";" << std::endl;
    header << "const unsigned int BANK_ID_MASK = (1 << BANK_ID_NBITS) - 1" << ";" << std::endl;
    header << "const unsigned int TILE_SIZE = " << this->out_buffer_len_ << ";" << std::endl;
    header << "const unsigned int BANK_SIZE = TILE_SIZE / NUM_BANK" << ";" << std::endl;
    header << "const unsigned int ARBITER_LATENCY = 6" << ";" << std::endl;
    header << "const unsigned int FWD_DISTANCE = ARBITER_LATENCY + 1" << ";" << std::endl;

    // Data types
    header << "typedef unsigned int INDEX_T;" << std::endl;
    header << "typedef " << this->val_t_str_ << " VAL_T;" << std::endl;

    header <<  "typedef struct {"
                  "INDEX_T index; VAL_T data;"
                "}" << " DIT_T;" << std::endl;

    header <<  "typedef struct {"
                  "INDEX_T indexpkt[PACKET_SIZE];"
                  "VAL_T datapkt[PACKET_SIZE];"
                "}" << " PACKED_DWI_T;" << std::endl;

    header <<  "template<typename index_t>"
                "struct rd_req {"
                  "bool valid;"
                  "bool zero;"
                  "index_t addr;"
                "};" << std::endl;

    header <<  "template<typename data_t>"
                "struct rd_resp {"
                  "bool valid;"
                  "data_t data;"
                "};" << std::endl;

    header <<  "template<typename data_t, typename index_t>"
                "struct wr_req {"
                  "data_t data;"
                  "index_t addr;"
                "};" << std::endl;

    header <<  "template<typename index_t>"
                "struct arbiter_result {"
                  "ap_uint<PE_ID_NBITS> virtual_port_id;"
                  "bool bank_idle;"
                "};" << std::endl;

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


template<typename matrix_data_t, typename vector_data_t, typename val_index_t>
void SpMSpVModule<matrix_data_t, vector_data_t, val_index_t>::generate_kernel_ini() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream ini(graphblas::proj_folder_name + "/" + this->kernel_name_ + ".ini");

    // memory channel connectivity
    ini << "[connectivity]" << std::endl;

    // allocate matrix on DDR1
    ini << "sp=kernel_spmspv_1.mat_dwi_ddr:DDR[1]" << std::endl;
    ini << "sp=kernel_spmspv_1.mat_idxptr_ddr:DDR[1]" << std::endl;
    ini << "sp=kernel_spmspv_1.mat_tileptr_ddr:DDR[1]" << std::endl;

    // allocate others on DDR0
    ini << "sp=kernel_spmspv_1.vec_dit_ddr:DDR[0]" << std::endl;
    if (this->use_mask_) {
        ini << "sp=kernel_spmspv_1.mask_ddr:DDR[0]" << std::endl;
    }
    ini << "sp=kernel_spmspv_1.result_ddr:DDR[0]" << std::endl;

    // enable retiming
    ini << "[vivado]" << std::endl;
    ini << "prop=run.__KERNEL__.{STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS}={-retiming}" << std::endl;

    ini.close();
}


template<typename matrix_data_t, typename vector_data_t, typename val_index_t>
void SpMSpVModule<matrix_data_t, vector_data_t, val_index_t>::load_and_format_matrix(CSCMatrix<float> const &csc_matrix_float) {
    this->csc_matrix_float_ = csc_matrix_float;
    this->csc_matrix_ = graphblas::io::csc_matrix_convert_from_float<matrix_data_t>(csc_matrix_float);
    SpMSpVDataFormatter<matrix_data_t,index_t,packet_t>
        formatter(this->csc_matrix_);
    formatter.format(this->out_buffer_len_,graphblas::pack_size);
    this->num_row_partitions_ = formatter.num_row_partitions();
    this->num_packets_ = formatter.num_packets_total();

    this->channel_packets_.resize(this->num_packets_);
    this->channel_indptr_.resize((this->get_num_cols() + 1) * this->num_row_partitions_);
    this->channel_partptr_.resize(this->num_row_partitions_ + 1);

    // read packets from formatter
    for (size_t i = 0; i < this->num_packets_; i++) {
        this->channel_packets_[i] = formatter.get_formatted_packet(i);
    }

    // read indptr from formatter
    for (size_t i = 0; i < (this->get_num_cols() + 1) * this->num_row_partitions_; i++) {
        this->channel_indptr_[i] = formatter.get_formatted_indptr(i);
    }

    // read partptr from formatter
    for (size_t i = 0; i < this->num_row_partitions_ + 1; i++) {
        this->channel_partptr_[i] = formatter.get_formatted_partptr(i);
    }

    // prepare output memory
    this->results_.resize(this->get_num_rows() + 1);

    std::fill(this->results_.begin(), this->results_.end(), (val_index_t){0,0});
}


template<typename matrix_data_t, typename vector_data_t, typename val_index_t>
void SpMSpVModule<matrix_data_t, vector_data_t, val_index_t>::send_matrix_host_to_device() {
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
        sizeof(index_t) * (this->get_num_cols() + 1) * this->num_row_partitions_ ,
        &channel_indptr_ext,
        &err));
    OCL_CHECK(err, this->channel_partptr_buf = cl::Buffer(this->context_,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(index_t) * (this->num_row_partitions_ + 1),
        &channel_partptr_ext,
        &err));

    // Handle results
    cl_mem_ext_ptr_t results_ext;

    results_ext.obj = this->results_.data();
    results_ext.param = 0;
    results_ext.flags = graphblas::DDR[0];

    // Allocate memory on the FPGA
    OCL_CHECK(err, this->results_buf = cl::Buffer(this->context_,
        CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(val_index_t) * (this->get_num_rows() + 1),
        &results_ext,
        &err));


    // Set arguments
    /*
      argidx (argidx wn mask):
      0      (0)             : sparse vector[with head]
      1      (1)             : matrix packets       (does not change over iteration)
      2      (2)             : matrix indptr        (does not change over iteration)
      3      (3)             : matrix partptr       (does not change over iteration)
      4      (-)             : dense mask
      5      (4)             : sparse results[with head]
      6      (5)             : number of columns    (does not change over iteration)
      7      (6)             : number of partitions (does not change over iteration)
    */
    size_t arg_idx = 1; // the first argument (arg_idx = 0) is the sparse vector
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->channel_packets_buf));
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->channel_indptr_buf));
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->channel_partptr_buf));

    if (this->use_mask_) {
        this->arg_idx_mask_ = arg_idx; // mask is right before results
        arg_idx++;
    }

    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->results_buf));
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->csc_matrix_.num_cols));
    OCL_CHECK(err, err = this->kernel_.setArg(arg_idx++, this->num_row_partitions_));

    // Send data to device
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({
                                this->channel_packets_buf,
                                this->channel_indptr_buf,
                                this->channel_partptr_buf,
                                }, 0 /* 0 means from host*/)
    );
    this->command_queue_.finish();
}


template<typename matrix_data_t, typename vector_data_t, typename val_index_t>
void SpMSpVModule<matrix_data_t, vector_data_t, val_index_t>::send_vector_host_to_device(aligned_val_index_t &vector) {
    cl_int err;

    // copy the input vector
    this->vector_.assign(vector.begin(), vector.end());
    std::cout << "[INFO send_vector_host_to_device] : external vector copied" << std::endl << std::flush;
    std::cout << "[INFO send_vector_host_to_device] : external vector size : " << vector.size() << std::endl << std::flush;
    index_t vector_nnz_cnt = vector[0].index;

    // Handle vector
    cl_mem_ext_ptr_t vector_ext;

    vector_ext.obj = this->vector_.data();
    vector_ext.param = 0;
    vector_ext.flags = graphblas::DDR[0];

    OCL_CHECK(err, this->vector_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(val_index_t) * (vector_nnz_cnt + 1),
                &vector_ext,
                &err));
    std::cout << "[INFO send_vector_host_to_device] : memory allocated on FPGA" << std::endl << std::flush;
    // set argument
    OCL_CHECK(err, err = this->kernel_.setArg(0, this->vector_buf));

    // Send data to device
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->vector_buf}, 0));
    this->command_queue_.finish();
    std::cout << "[INFO send_vector_host_to_device] : finished" << std::endl << std::flush;
}


template<typename matrix_data_t, typename vector_data_t, typename val_index_t>
void SpMSpVModule<matrix_data_t, vector_data_t, val_index_t>::send_mask_host_to_device(aligned_val_t &mask) {
    cl_int err;

    this->mask_.assign(mask.begin(), mask.end());

    cl_mem_ext_ptr_t mask_ext;

    mask_ext.obj = this->mask_.data();
    mask_ext.param = 0;
    mask_ext.flags = graphblas::DDR[0];

    OCL_CHECK(err, this->mask_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * this->csc_matrix_.num_rows,
                &mask_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(this->arg_idx_mask_, this->mask_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->mask_buf}, 0));
    this->command_queue_.finish();
}


template<typename matrix_data_t, typename vector_data_t, typename val_index_t>
void SpMSpVModule<matrix_data_t, vector_data_t, val_index_t>::run() {
    cl_int err;
    OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->kernel_));
    this->command_queue_.finish();
}

// reference without mask
template<typename matrix_data_t, typename vector_data_t, typename val_index_t>
graphblas::aligned_sparse_float_vec_t
SpMSpVModule<matrix_data_t, vector_data_t, val_index_t>::compute_reference_results(graphblas::aligned_sparse_float_vec_t &vector) {
    // measure dimensions
    unsigned int vec_nnz_total = vector[0].index;
    unsigned int num_rows = this->csc_matrix_.num_rows;

    // create result container
    aligned_sparse_float_vec_t reference_results;

    // local buffer for dense results
    std::vector<vector_data_t> output_buffer(num_rows,0);

    // indices of active columns are stored in vec_idx
    // number of active columns = vec_nnz_total
    // loop over all active columns
    for (unsigned int active_colid = 0; active_colid < vec_nnz_total; active_colid++) {
        index_t current_colid = vector[active_colid + 1].index;
        // slice out the current column out of the active columns
        index_t col_id_start = this->csc_matrix_.adj_indptr[current_colid];
        index_t col_id_end = this->csc_matrix_.adj_indptr[current_colid + 1];
        index_t current_collen = col_id_end - col_id_start;
        matrix_data_t current_col[current_collen];
        index_t current_row_ids[current_collen];
        for (unsigned int i = 0; i < current_collen; i++) {
            current_col[i] = this->csc_matrix_.adj_data[i + col_id_start];
            current_row_ids[i] = this->csc_matrix_.adj_indices[i + col_id_start];
        }

        // loop over all nnzs in the current column
        for (unsigned int mat_element_id = 0; mat_element_id < current_collen; mat_element_id++) {
            index_t current_row_id = current_row_ids[mat_element_id];
            vector_data_t nnz_from_mat = current_col[mat_element_id];
            vector_data_t nnz_from_vec = vector[active_colid + 1].val;
            switch (this->semiring_) {
                case graphblas::kMulAdd:
                    output_buffer[current_row_id] += nnz_from_mat * nnz_from_vec;
                    break;
                case graphblas::kLogicalAndOr:
                    output_buffer[current_row_id] = output_buffer[current_row_id] || (nnz_from_mat && nnz_from_vec);
                    break;
                default:
                    std::cerr << "Invalid semiring" << std::endl;
                    break;
            }
        }
    }

    // checkout results
    index_t nnz_cnt = 0;
    reference_results.clear();
    for (size_t obid = 0; obid < num_rows; obid++) {
        if(output_buffer[obid]) {
            index_float_t a;
            a.val = output_buffer[obid];
            a.index = obid;
            reference_results.push_back(a);
            nnz_cnt ++;
        }
    }
    index_float_t reference_head;
    reference_head.val = 0;
    reference_head.index = nnz_cnt;
    reference_results.insert(reference_results.begin(),reference_head);
    return reference_results;
}


// reference with mask
template<typename matrix_data_t, typename vector_data_t, typename val_index_t>
graphblas::aligned_sparse_float_vec_t
SpMSpVModule<matrix_data_t, vector_data_t, val_index_t>::compute_reference_results(graphblas::aligned_sparse_float_vec_t &vector,
                                                                                   graphblas::aligned_dense_float_vec_t &mask) {
    // measure dimensions
    unsigned int vec_nnz_total = vector[0].index;
    unsigned int num_rows = this->csc_matrix_.num_rows;

    // create result container
    aligned_sparse_float_vec_t reference_results;

    // local buffer for dense results
    std::vector<vector_data_t> output_buffer(num_rows,0);

    // indices of active columns are stored in vec_idx
    // number of active columns = vec_nnz_total
    // loop over all active columns
    for (unsigned int active_colid = 0; active_colid < vec_nnz_total; active_colid++) {
        index_t current_colid = vector[active_colid + 1].index;
        // slice out the current column out of the active columns
        index_t col_id_start = this->csc_matrix_.adj_indptr[current_colid];
        index_t col_id_end = this->csc_matrix_.adj_indptr[current_colid + 1];
        index_t current_collen = col_id_end - col_id_start;
        matrix_data_t current_col[current_collen];
        index_t current_row_ids[current_collen];
        for (unsigned int i = 0; i < current_collen; i++) {
            current_col[i] = this->csc_matrix_.adj_data[i + col_id_start];
            current_row_ids[i] = this->csc_matrix_.adj_indices[i + col_id_start];
        }

        // loop over all nnzs in the current column
        for (unsigned int mat_element_id = 0; mat_element_id < current_collen; mat_element_id++) {
            index_t current_row_id = current_row_ids[mat_element_id];
            vector_data_t nnz_from_mat = current_col[mat_element_id];
            vector_data_t nnz_from_vec = vector[active_colid + 1].val;
            switch (this->semiring_) {
                case graphblas::kMulAdd:
                    output_buffer[current_row_id] += nnz_from_mat * nnz_from_vec;
                    break;
                case graphblas::kLogicalAndOr:
                    output_buffer[current_row_id] = output_buffer[current_row_id] || (nnz_from_mat && nnz_from_vec);
                    break;
                default:
                    std::cerr << "Invalid semiring" << std::endl;
                    break;
            }
        }
    }

    // checkout results
    index_t nnz_cnt = 0;
    reference_results.clear();
    for (size_t obid = 0; obid < num_rows; obid++) {
        bool do_write = false;
        switch (this->mask_type_) {
        case graphblas::kMaskWriteToZero:
            do_write = (mask[obid] == 0);
            break;
        case graphblas::kMaskWriteToOne:
            do_write = (mask[obid] != 0);
            break;
        default:
            std::cerr << "Invalid Mask Type" << std::endl;
            break;
        }
        if(do_write) {
            if(output_buffer[obid]) {
                graphblas::index_float_t a;
                a.val = output_buffer[obid];
                a.index = obid;
                reference_results.push_back(a);
                nnz_cnt ++;
            }
        }
    }
    index_float_t reference_head;
    reference_head.val = 0;
    reference_head.index = nnz_cnt;
    reference_results.insert(reference_results.begin(),reference_head);
    return reference_results;
}


} // namespace module
} // namespace graphblas

#endif // __GRAPHBLAS_SPMSPV_MODULE_H
