#ifndef GRAPHLILY_SPMV_MODULE_H_
#define GRAPHLILY_SPMV_MODULE_H_

#include <cstdint>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>

#include "xcl2.hpp"

#include "graphlily/global.h"
#include "graphlily/io/data_loader.h"
#include "graphlily/io/data_formatter.h"
#include "graphlily/module/base_module.h"


using graphlily::io::CSRMatrix;
using graphlily::io::CPSRMatrix;
using graphlily::io::csr2cpsr;


namespace {
template<typename virtual_packed_t, uint32_t virtual_pack_size, typename packed_t, uint32_t pack_size>
inline packed_t _slice(const virtual_packed_t &in, uint32_t start_idx) {
    packed_t out;
    for (size_t i = 0; i < pack_size; i++) {
        out.data[i] = in.data[i + start_idx];
    }
    return out;
}
}  // namespace


namespace graphlily {
namespace module {

template<typename matrix_data_t, typename vector_data_t>
class SpMVModule : public BaseModule {
private:
    /*! \brief The mask type */
    graphlily::MaskType mask_type_;
    /*! \brief The semiring */
    graphlily::SemiringType semiring_;
    /*! \brief The number of channels of the kernel */
    uint32_t num_channels_;
    /*! \brief The number of channels for sub-kernel 0 */
    uint32_t num_channels_sk0_;
    /*! \brief The number of channels for sub-kernel 1 */
    uint32_t num_channels_sk1_;
    /*! \brief The number of channels for sub-kernel 2 */
    uint32_t num_channels_sk2_;
    /*! \brief The length of output buffer of the kernel */
    uint32_t out_buf_len_;
    /*! \brief The number of row partitions */
    uint32_t num_row_partitions_;
    /*! \brief The length of vector buffer of the kernel */
    uint32_t vec_buf_len_;
    /*! \brief The number of column partitions */
    uint32_t num_col_partitions_;

    using val_t = vector_data_t;
    using packed_val_t = typename CPSRMatrix<val_t, graphlily::pack_size>::packed_val_t;
    using packed_idx_t = typename CPSRMatrix<val_t, graphlily::pack_size>::packed_idx_t;
    using mat_pkt_t = struct {packed_idx_t indices; packed_val_t vals;};
    using partition_indptr_t = struct {graphlily::idx_t start; unsigned max_nnz; packed_idx_t nnz;};

    using aligned_idx_t = std::vector<graphlily::idx_t, aligned_allocator<graphlily::idx_t>>;
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
    /*! \brief The kernel results */
    aligned_dense_vec_t results_;
    /*! \brief The sparse matrix using float data type in CSR format */
    CSRMatrix<float> csr_matrix_float_;
    /*! \brief The sparse matrix in CSR format */
    CSRMatrix<val_t> csr_matrix_;

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

public:

    SpMVModule(uint32_t num_channels,
               uint32_t out_buf_len,
               uint32_t vec_buf_len) : BaseModule("overlay") {
        this->_check_data_type();
        this->num_channels_ = num_channels;
        this->out_buf_len_ = out_buf_len;
        this->vec_buf_len_ = vec_buf_len;
        // TODO: parameterize channel allocation
        // ! will not work if num_channels != 16
        this->num_channels_sk0_ = 4;
        this->num_channels_sk1_ = 6;
        this->num_channels_sk2_ = 6;
    }

    /*Overlay argument list:
    * (H = num_hbm_channels for this spmv sub-kernel)
    * Index       Argument                     used in this module?
    * 0 ~ H-1     matrix for spmv              y
    * H+2         row partition id
    * H+3         partition length
    * H+4         number of column partitions
    * H+5         number of total partitions
    * H+6         semiring
    * H+7         zero value of this semiring
    */
    void set_unused_args() override {
        // ** no unused args in split-kernel case
        // Set unused arguments for SpMSpV
        // for (uint32_t i = this->num_channels_ + 4; i < this->num_channels_ + 10; i++) {
        //     this->kernel_.setArg(i, cl::Buffer(this->context_, 0, 4));
        // }
        // // Set unused scalar arguments
        // this->kernel_.setArg(this->num_channels_ + 15, (unsigned)NULL);
        // // To avoid runtime error of invalid scalar argument size
        // if (!(std::is_same<vector_data_t, unsigned>::value || std::is_same<vector_data_t, float>::value)) {
        //     this->kernel_.setArg(this->num_channels_ + 16, (long long)NULL);
        // } else {
        //     this->kernel_.setArg(this->num_channels_ + 16, (unsigned)NULL);
        // }
        // if (this->mask_type_ == graphlily::kNoMask) {
        //     this->kernel_.setArg(this->num_channels_ + 1, cl::Buffer(this->context_, 0, 4));
        //     this->kernel_.setArg(this->num_channels_ + 2, cl::Buffer(this->context_, 0, 4));
        // }
    }

    void set_mode() override {
        // ** no mode needs to be set in split-kernel case
        // this->kernel_.setArg(this->num_channels_ + 14, 1);  // 1 is SpMV
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
     * \param skip_empty_rows Whether skip empty rows.
     */
    void load_and_format_matrix(CSRMatrix<float> const &csr_matrix_float, bool skip_empty_rows);

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

    /*!
     * \brief Bind the mask buffer to an existing buffer.
     */
    void bind_mask_buf(cl::Buffer src_buf);

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
    graphlily::aligned_dense_float_vec_t
    compute_reference_results(graphlily::aligned_dense_float_vec_t &vector);

    /*!
     * \brief Compute reference results.
     * \param vector The dense vector.
     * \param mask The mask.
     * \return The reference results.
     */
    graphlily::aligned_dense_float_vec_t
    compute_reference_results(graphlily::aligned_dense_float_vec_t &vector,
                              graphlily::aligned_dense_float_vec_t &mask);
};


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::_check_data_type() {
    assert((std::is_same<matrix_data_t, vector_data_t>::value));
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::load_and_format_matrix(
        CSRMatrix<float> const &csr_matrix_float,
        bool skip_empty_rows) {
    this->csr_matrix_float_ = csr_matrix_float;
    this->csr_matrix_ = graphlily::io::csr_matrix_convert_from_float<val_t>(csr_matrix_float);
    this->num_row_partitions_ = (this->csr_matrix_.num_rows + this->out_buf_len_ - 1) / this->out_buf_len_;
    this->num_col_partitions_ = (this->csr_matrix_.num_cols + this->vec_buf_len_ - 1) / this->vec_buf_len_;
    size_t num_partitions = this->num_row_partitions_ * this->num_col_partitions_;
    const uint32_t virtual_pack_size = graphlily::pack_size * graphlily::spmv_row_interleave_factor;
    CPSRMatrix<val_t, virtual_pack_size> cpsr_matrix = csr2cpsr<val_t, virtual_pack_size>(
        this->csr_matrix_,
        graphlily::idx_marker,
        this->out_buf_len_,
        this->vec_buf_len_,
        this->num_channels_,
        skip_empty_rows);
    // TODO: remove row-interleaving since HiSparse(FX) doesn't support it
    // Data types to support virtual pack size
    using virtual_packed_idx_t = typename CPSRMatrix<val_t, virtual_pack_size>::packed_idx_t;
    using virtual_packed_val_t = typename CPSRMatrix<val_t, virtual_pack_size>::packed_val_t;
    using virtual_partition_indptr_t = struct {graphlily::idx_t start; unsigned max_nnz; virtual_packed_idx_t nnz;};
    using aligned_virtual_packed_idx_t = std::vector<virtual_packed_idx_t, aligned_allocator<virtual_packed_idx_t>>;
    using aligned_virtual_packed_val_t = std::vector<virtual_packed_val_t, aligned_allocator<virtual_packed_val_t>>;
    using aligned_virtual_partition_indptr_t = std::vector<virtual_partition_indptr_t, aligned_allocator<virtual_partition_indptr_t>>;
    // Intermediate data of the above data types
    std::vector<aligned_virtual_packed_idx_t> channel_indices(this->num_channels_);
    std::vector<aligned_virtual_packed_val_t> channel_vals(this->num_channels_);
    std::vector<aligned_virtual_partition_indptr_t> channel_partition_indptr(this->num_channels_);
    this->channel_packets_.resize(this->num_channels_);
    for (size_t c = 0; c < this->num_channels_; c++) {
        channel_partition_indptr[c].resize(num_partitions);
        channel_partition_indptr[c][0].start = 0;
    }
    // Iterate the channels
    for (size_t c = 0; c < this->num_channels_; c++) {
        for (size_t j = 0; j < this->num_row_partitions_; j++) {
            for (size_t i = 0; i < this->num_col_partitions_; i++) {
                auto indices_partition = cpsr_matrix.get_packed_indices(j, i, c);
                channel_indices[c].insert(channel_indices[c].end(),
                    indices_partition.begin(), indices_partition.end());
                auto vals_partition = cpsr_matrix.get_packed_data(j, i, c);
                channel_vals[c].insert(channel_vals[c].end(),
                    vals_partition.begin(), vals_partition.end());
                assert(indices_partition.size() == vals_partition.size());
                auto indptr_partition = cpsr_matrix.get_packed_indptr(j, i, c);
                if (!((j == (this->num_row_partitions_ - 1)) && (i == (this->num_col_partitions_ - 1)))) {
                    channel_partition_indptr[c][j*this->num_col_partitions_ + i + 1].start =
                        channel_partition_indptr[c][j*this->num_col_partitions_ + i].start
                        + indices_partition.size();
                }
                channel_partition_indptr[c][j*this->num_col_partitions_ + i].nnz = indptr_partition.back();
                uint32_t max_nnz = *std::max_element(indptr_partition.back().data,
                                                     indptr_partition.back().data + virtual_pack_size);
                assert(indices_partition.size() == max_nnz);
                channel_partition_indptr[c][j*this->num_col_partitions_ + i].max_nnz = max_nnz;
            }
        }
        assert(channel_indices[c].size() == channel_vals[c].size());
        this->channel_packets_[c].resize(num_partitions*(1+graphlily::spmv_row_interleave_factor)
                                         + channel_indices[c].size()*graphlily::spmv_row_interleave_factor);
        // partition indptr
        for (size_t i = 0; i < num_partitions; i++) {
            this->channel_packets_[c][i*(1+graphlily::spmv_row_interleave_factor)].indices.data[0] =
                channel_partition_indptr[c][i].start * graphlily::spmv_row_interleave_factor;
            this->channel_packets_[c][i*(1+graphlily::spmv_row_interleave_factor)].indices.data[1] =
                channel_partition_indptr[c][i].max_nnz * graphlily::spmv_row_interleave_factor;
            for (size_t k = 0; k < graphlily::spmv_row_interleave_factor; k++) {
                this->channel_packets_[c][i*(1+graphlily::spmv_row_interleave_factor) + 1 + k].indices =
                    _slice<virtual_packed_idx_t, virtual_pack_size, packed_idx_t, graphlily::pack_size>(
                        channel_partition_indptr[c][i].nnz, k*graphlily::pack_size);
            }
        }
        // matrix indices and vals
        uint32_t offset = num_partitions*(1+graphlily::spmv_row_interleave_factor);
        for (size_t i = 0; i < channel_indices[c].size(); i++) {
            for (size_t k = 0; k < graphlily::spmv_row_interleave_factor; k++) {
                uint32_t ii = i * graphlily::spmv_row_interleave_factor + k;
                this->channel_packets_[c][offset + ii].indices =
                    _slice<virtual_packed_idx_t, virtual_pack_size, packed_idx_t, graphlily::pack_size>(
                        channel_indices[c][i], k*graphlily::pack_size);
                this->channel_packets_[c][offset + ii].vals =
                    _slice<virtual_packed_val_t, virtual_pack_size, packed_val_t, graphlily::pack_size>(
                        channel_vals[c][i], k*graphlily::pack_size);
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
    std::cout << "Start to allocate memory on "
              << this->num_channels_ << " channels" << std::endl;
    // Handle channel_packets
    // cl_mem_ext_ptr_t channel_packets_ext[this->num_channels_];
    std::vector<cl_mem_ext_ptr_t> channel_packets_ext;
    channel_packets_ext.resize(this->num_channels_);
    this->channel_packets_buf.resize(this->num_channels_);
    for (unsigned c = 0; c < this->num_channels_; c++) {
        channel_packets_ext[c].obj = this->channel_packets_[c].data();
        channel_packets_ext[c].param = 0;
        channel_packets_ext[c].flags = graphlily::HBM[c];
        // channel_packets_ext[c].flags = 0x80000000 + c;
        size_t channel_packets_size = sizeof(mat_pkt_t) * this->channel_packets_[c].size();
        // std::cout << "channel_packets_size: " << channel_packets_size << std::endl;
        if (channel_packets_size >= 256 * 1000 * 1000) {
            std::cout << "The capcity of one HBM channel is 256 MB" << std::endl;
            exit(EXIT_FAILURE);
        }
        OCL_CHECK(err, this->channel_packets_buf[c] = cl::Buffer(this->context_,
            CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
            channel_packets_size,
            &channel_packets_ext[c],
            &err));
    }

    // for debug
    for (unsigned c = 0; c < this->num_channels_; c++) {
        void* buf_host_ptr;
        this->channel_packets_buf[c].getInfo(CL_MEM_HOST_PTR, &buf_host_ptr);
        printf("channel_packets_buf[%2d]: host pointer %p; channel_packets_ext[%2d]: obj %p, flags 0x%0x",
               c, buf_host_ptr,
               c, (void*)channel_packets_ext[c].obj, channel_packets_ext[c].flags);
        std::cout << std::endl;
    }

    // Handle results
    cl_mem_ext_ptr_t results_ext;
    results_ext.obj = this->results_.data();
    results_ext.param = 0;
    results_ext.flags = graphlily::HBM[22];
    OCL_CHECK(err, this->results_buf = cl::Buffer(this->context_,
        CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(val_t) * this->csr_matrix_.num_rows,
        &results_ext,
        &err));
    // Set arguments
    unsigned ch_offset = 0;
    for (size_t c = 0; c < this->num_channels_sk0_; c++) {
        std::cout << "Setting SK0 argument " << c << " to HBM " << c + ch_offset << std::endl;
        OCL_CHECK(err, err = this->spmv_sk0_.setArg(c, this->channel_packets_buf[c + ch_offset]));
    }
    ch_offset += this->num_channels_sk0_;
    for (size_t c = 0; c < this->num_channels_sk1_; c++) {
        std::cout << "Setting SK1 argument " << c << " to HBM " << c + ch_offset << std::endl;
        OCL_CHECK(err, err = this->spmv_sk1_.setArg(c, this->channel_packets_buf[c + ch_offset]));
    }
    ch_offset += this->num_channels_sk1_;
    for (size_t c = 0; c < this->num_channels_sk2_; c++) {
        std::cout << "Setting SK2 argument " << c << " to HBM " << c + ch_offset << std::endl;
        OCL_CHECK(err, err = this->spmv_sk2_.setArg(c, this->channel_packets_buf[c + ch_offset]));
    }
    unsigned num_partitions = this->num_row_partitions_ * this->num_col_partitions_;
    val_t val_zero;
    switch (this->semiring_.op) {
    case kMulAdd:
        val_zero = 0;
        break;
    case kLogicalAndOr:
        val_zero = 0;
        break;
    case kAddMin:
        val_zero = UFIXED_INF;
        break;
    default:
        val_zero = 0;
        break;
    }
    unsigned zero = graphlily::pack_raw_bits_to_uint(val_zero);
    OCL_CHECK(err, err = this->spmv_sk0_.setArg(this->num_channels_sk0_ + 4, (unsigned)this->num_col_partitions_));
    OCL_CHECK(err, err = this->spmv_sk0_.setArg(this->num_channels_sk0_ + 5, (unsigned)num_partitions));
    OCL_CHECK(err, err = this->spmv_sk0_.setArg(this->num_channels_sk0_ + 6, (char)this->semiring_.op));
    OCL_CHECK(err, err = this->spmv_sk0_.setArg(this->num_channels_sk0_ + 7, (unsigned)zero));
    OCL_CHECK(err, err = this->spmv_sk1_.setArg(this->num_channels_sk1_ + 4, (unsigned)this->num_col_partitions_));
    OCL_CHECK(err, err = this->spmv_sk1_.setArg(this->num_channels_sk1_ + 5, (unsigned)num_partitions));
    OCL_CHECK(err, err = this->spmv_sk1_.setArg(this->num_channels_sk1_ + 6, (char)this->semiring_.op));
    OCL_CHECK(err, err = this->spmv_sk1_.setArg(this->num_channels_sk1_ + 7, (unsigned)zero));
    OCL_CHECK(err, err = this->spmv_sk2_.setArg(this->num_channels_sk2_ + 4, (unsigned)this->num_col_partitions_));
    OCL_CHECK(err, err = this->spmv_sk2_.setArg(this->num_channels_sk2_ + 5, (unsigned)num_partitions));
    OCL_CHECK(err, err = this->spmv_sk2_.setArg(this->num_channels_sk2_ + 6, (char)this->semiring_.op));
    OCL_CHECK(err, err = this->spmv_sk2_.setArg(this->num_channels_sk2_ + 7, (unsigned)zero));
    OCL_CHECK(err, err = this->spmv_result_drain_.setArg(0, this->results_buf));

    // for (size_t c = 0; c < this->num_channels_; c++) {
    //     OCL_CHECK(err, err = this->kernel_.setArg(c, this->channel_packets_buf[c]));
    // }
    // OCL_CHECK(err, err = this->kernel_.setArg(this->num_channels_ + 3, this->results_buf));
    // OCL_CHECK(err, err = this->kernel_.setArg(this->num_channels_ + 10, this->csr_matrix_.num_rows));
    // OCL_CHECK(err, err = this->kernel_.setArg(this->num_channels_ + 11, this->csr_matrix_.num_cols));
    // OCL_CHECK(err, err = this->kernel_.setArg(this->num_channels_ + 12, (char)this->semiring_.op));
    // OCL_CHECK(err, err = this->kernel_.setArg(this->num_channels_ + 13, (char)this->mask_type_));

    // Send data to device
    for (size_t c = 0; c < this->num_channels_; c++) {
        OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects(
            {this->channel_packets_buf[c]}, 0 /* 0 means from host to device */ ));
    }
    this->command_queue_.finish();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_vector_host_to_device(aligned_dense_vec_t &vector) {
    this->vector_.assign(vector.begin(), vector.end());
    cl_mem_ext_ptr_t vector_ext;
    vector_ext.obj = this->vector_.data();
    vector_ext.param = 0;
    vector_ext.flags = graphlily::HBM[20];
    cl_int err;
    OCL_CHECK(err, this->vector_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(val_t) * this->csr_matrix_.num_cols,
                &vector_ext,
                &err));
    // OCL_CHECK(err, err = this->kernel_.setArg(this->num_channels_ + 0, this->vector_buf));
    OCL_CHECK(err, err = this->spmv_vector_loader_.setArg(0, this->vector_buf));
    OCL_CHECK(err, err = this->spmv_vector_loader_.setArg(1, (unsigned)this->vector_.size()));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->vector_buf}, 0));
    this->command_queue_.finish();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_mask_host_to_device(aligned_dense_vec_t &mask) {
    // TODO: support mask for split-kernel spmv
    // this->mask_.assign(mask.begin(), mask.end());
    // cl_mem_ext_ptr_t mask_ext;
    // mask_ext.obj = this->mask_.data();
    // mask_ext.param = 0;
    // mask_ext.flags = graphlily::HBM[21];
    // cl_int err;
    // OCL_CHECK(err, this->mask_buf = cl::Buffer(this->context_,
    //             CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
    //             sizeof(val_t) * this->csr_matrix_.num_rows,
    //             &mask_ext,
    //             &err));
    // OCL_CHECK(err, err = this->kernel_.setArg(this->num_channels_ + 1, this->mask_buf));
    // OCL_CHECK(err, err = this->kernel_.setArg(this->num_channels_ + 2, this->mask_buf));
    // OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->mask_buf}, 0));
    // this->command_queue_.finish();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::bind_mask_buf(cl::Buffer src_buf) {
    // TODO: support mask for split-kernel spmv
    // this->mask_buf = src_buf;
    // this->kernel_.setArg(this->num_channels_ + 1, this->mask_buf);
    // this->kernel_.setArg(this->num_channels_ + 2, this->mask_buf);
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::run() {
    cl_int err;
    // OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->kernel_));
    // this->command_queue_.finish();

    size_t rows_per_ch_in_last_row_part;
    if (this->csr_matrix_.num_rows % this->out_buf_len_ == 0) {
        rows_per_ch_in_last_row_part = this->out_buf_len_ / this->num_channels_;
    } else {
        rows_per_ch_in_last_row_part = this->csr_matrix_.num_rows % this->out_buf_len_ / this->num_channels_;
    }
    for (size_t row_part_id = 0; row_part_id < this->num_row_partitions_; row_part_id++) {
        unsigned part_len = this->out_buf_len_ / this->num_channels_;
        if (row_part_id == this->num_row_partitions_ - 1) {
            part_len = rows_per_ch_in_last_row_part;
        }
        // std::cout << "INFO : SpMV Kernel Started: row partition " << row_part_id
                //   << " with " << part_len << " rows per cluster" << std::endl;
        OCL_CHECK(err, err = this->spmv_sk0_.setArg(this->num_channels_sk0_ + 2, (unsigned)row_part_id));
        OCL_CHECK(err, err = this->spmv_sk0_.setArg(this->num_channels_sk0_ + 3, (unsigned)part_len));
        OCL_CHECK(err, err = this->spmv_sk1_.setArg(this->num_channels_sk1_ + 2, (unsigned)row_part_id));
        OCL_CHECK(err, err = this->spmv_sk1_.setArg(this->num_channels_sk1_ + 3, (unsigned)part_len));
        OCL_CHECK(err, err = this->spmv_sk2_.setArg(this->num_channels_sk2_ + 2, (unsigned)row_part_id));
        OCL_CHECK(err, err = this->spmv_sk2_.setArg(this->num_channels_sk2_ + 3, (unsigned)part_len));
        OCL_CHECK(err, err = this->spmv_result_drain_.setArg(1, (unsigned)row_part_id));

        OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->spmv_vector_loader_));
        OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->spmv_sk0_));
        OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->spmv_sk1_));
        OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->spmv_sk2_));
        OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->spmv_result_drain_));
        this->command_queue_.finish();
        // std::cout << "INFO : SpMV Kernel Finished: row partition " << row_part_id << std::endl;
    }
    // std::cout << "INFO : SpMV kernel complete!" << std::endl;
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
graphlily::aligned_dense_float_vec_t
SpMVModule<matrix_data_t, vector_data_t>::compute_reference_results(aligned_dense_float_vec_t &vector) {
    aligned_dense_float_vec_t reference_results(this->csr_matrix_.num_rows);
    std::fill(reference_results.begin(), reference_results.end(), this->semiring_.zero);
    switch (this->semiring_.op) {
        case graphlily::kMulAdd:
            SPMV(reference_results[row_idx] += this->csr_matrix_float_.adj_data[i] * vector[idx]);
            break;
        case graphlily::kLogicalAndOr:
            SPMV(reference_results[row_idx] = reference_results[row_idx]
                || (this->csr_matrix_float_.adj_data[i] && vector[idx]));
            break;
        case graphlily::kAddMin:
            SPMV(reference_results[row_idx] = std::min(reference_results[row_idx],
                this->csr_matrix_float_.adj_data[i] + vector[idx]));
            break;
        default:
            std::cerr << "Invalid semiring" << std::endl;
            break;
    }
    return reference_results;
}


template<typename matrix_data_t, typename vector_data_t>
graphlily::aligned_dense_float_vec_t SpMVModule<matrix_data_t, vector_data_t>::compute_reference_results(
        graphlily::aligned_dense_float_vec_t &vector,
        graphlily::aligned_dense_float_vec_t &mask) {
    graphlily::aligned_dense_float_vec_t reference_results = this->compute_reference_results(vector);
    if (this->mask_type_ == graphlily::kMaskWriteToZero) {
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
}  // namespace graphlily

#endif  // GRAPHLILY_SPMV_MODULE_H_
