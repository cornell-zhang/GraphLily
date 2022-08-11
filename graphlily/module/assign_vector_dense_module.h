#ifndef GRAPHLILY_ASSIGN_VECTOR_DENSE_MODULE_H_
#define GRAPHLILY_ASSIGN_VECTOR_DENSE_MODULE_H_

#include <cstdint>
#include <vector>
#include <fstream>
#include <chrono>

#include "xcl2.hpp"

#include "graphlily/global.h"
#include "graphlily/module/base_module.h"


namespace graphlily {
namespace module {

template<typename vector_data_t>
class AssignVectorDenseModule : public BaseModule {
private:
    using packed_val_t = struct {vector_data_t data[graphlily::pack_size];};
    using aligned_dense_vec_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;

    /*! \brief The mask type */
    graphlily::MaskType mask_type_;
    /*! \brief Internal copy of mask */
    aligned_dense_vec_t mask_;
    /*! \brief Internal copy of inout */
    aligned_dense_vec_t inout_;
    /*! \brief The offset of spmv(VL, RD), spmspv, and apply kernel
               equals to 4 + spmspv HBM channels */
    uint32_t overlay_arg_offset_;

public:
    // Device buffers
    cl::Buffer mask_buf;
    cl::Buffer inout_buf;

public:
    AssignVectorDenseModule() : BaseModule() {
        this->overlay_arg_offset_ = 4 + graphlily::spmspv_num_hbm_channels;
    }

    /* Overlay (VL, RD of SpMV, SpMSpV, and apply) argument list:
    *  (k = 4 + 1 * SpMSpV HBM channels)
    * Index       Argument                              Used in this module?
    * 0           vector for spmv                       y
    * 1           mask for spmv (read port)             y
    * 2           mask for spmv (write port)            y
    * 3           output for spmv                       n
    *
    * 4~k-1       matrix for spmspv start from HBM[23]  n
    *
    * k+0         vector for spmspv                     n
    * k+1         mask for spmspv                       n
    * k+2         output for spmspv                     n
    *
    * k+3         number of rows                        n
    * k+4         number of columns                     n
    * k+5         semiring operation type               n
    *
    * k+6         mask type                             y
    * k+7         overlay mode select                   y
    * k+8         apply vector length                   y
    * k+9         apply input value or semiring zero    y
    * k+10        spmv RD row partition index           n
    */
    void set_unused_args() override {
        // Set unused arguments for SpMV
        this->spmv_vl_rd_spmspv_apply_.setArg(3, cl::Buffer(this->context_, 0, 4));
        // Set unused arguments for SpMSpV
        for (size_t i = 4; i <= this->overlay_arg_offset_ + 2; ++i) {
            this->spmv_vl_rd_spmspv_apply_.setArg(i, cl::Buffer(this->context_, 0, 4));
        }
        // Set unused scalar arguments
        this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 3, (unsigned)NULL);
        this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 4, (unsigned)NULL);
        this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 5, (char)NULL);
        this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 10, (unsigned)NULL);
    }

    void set_mode() override {
        this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 7, 4);;  // 4 is kernel_assign_vector_dense
    }

    /*!
     * \brief Set the mask type.
     * \param mask_type The mask type.
     */
    void set_mask_type(graphlily::MaskType mask_type) {
        if (mask_type == graphlily::kNoMask) {
            std::cerr << "Please set the mask type" << std::endl;
            exit(EXIT_FAILURE);
        } else {
            this->mask_type_ = mask_type;
        }
    }

    /*!
     * \brief Send the mask from host to device.
     */
    void send_mask_host_to_device(aligned_dense_vec_t &mask);

    /*!
     * \brief Send the inout from host to device.
     */
    void send_inout_host_to_device(aligned_dense_vec_t &inout);

    /*!
     * \brief Bind the mask buffer to an existing buffer.
     */
    void bind_mask_buf(cl::Buffer src_buf) {
        this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 6, (char)this->mask_type_);
        this->mask_buf = src_buf;
        this->spmv_vl_rd_spmspv_apply_.setArg(0, this->mask_buf);
    }

    /*!
     * \brief Bind the inout buffer to an existing buffer.
     */
    void bind_inout_buf(cl::Buffer src_buf) {
        this->inout_buf = src_buf;
        // set both read and write ports
        this->spmv_vl_rd_spmspv_apply_.setArg(1, this->inout_buf);
        this->spmv_vl_rd_spmspv_apply_.setArg(2, this->inout_buf);
    }

    /*!
     * \brief Run the module.
     * \param len The length of the mask/inout vector.
     * \param val The value to be assigned to the inout vector.
     */
    void run(uint32_t len, vector_data_t val);

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
     * \brief Send the inout from device to host.
     * \return The inout.
     */
    aligned_dense_vec_t send_inout_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->inout_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->inout_;
    }

    /*!
     * \brief Compute reference results.
     * \param mask The mask vector.
     * \param inout The inout vector.
     * \param len The length of the mask/inout vector.
     * \param val The value to be assigned to the inout vector.
     */
    void compute_reference_results(graphlily::aligned_dense_float_vec_t &mask,
                                   graphlily::aligned_dense_float_vec_t &inout,
                                   uint32_t len,
                                   float val);
};


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::send_mask_host_to_device(aligned_dense_vec_t &mask) {
    this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 6, (char)this->mask_type_);
    this->mask_.assign(mask.begin(), mask.end());
    cl_mem_ext_ptr_t mask_ext;
    mask_ext.obj = this->mask_.data();
    mask_ext.param = 0;
    mask_ext.flags = graphlily::HBM[20];
    cl_int err;
    OCL_CHECK(err, this->mask_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * this->mask_.size(),
                &mask_ext,
                &err));
    OCL_CHECK(err, err = this->spmv_vl_rd_spmspv_apply_.setArg(0, this->mask_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->mask_buf}, 0));
    this->command_queue_.finish();
}


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::send_inout_host_to_device(aligned_dense_vec_t &inout) {
    this->inout_.assign(inout.begin(), inout.end());
    cl_mem_ext_ptr_t inout_ext;
    inout_ext.obj = this->inout_.data();
    inout_ext.param = 0;
    inout_ext.flags = graphlily::HBM[21];
    cl_int err;
    OCL_CHECK(err, this->inout_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * this->inout_.size(),
                &inout_ext,
                &err));
    // set both read and write ports
    OCL_CHECK(err, err = this->spmv_vl_rd_spmspv_apply_.setArg(1, this->inout_buf));
    OCL_CHECK(err, err = this->spmv_vl_rd_spmspv_apply_.setArg(2, this->inout_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->inout_buf}, 0));
    this->command_queue_.finish();
}


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::run(uint32_t len, vector_data_t val) {
    cl_int err;
    OCL_CHECK(err, err = this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 8, len));
    OCL_CHECK(err, err = this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 9, graphlily::pack_raw_bits_to_uint(val)));

    OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->spmv_vl_rd_spmspv_apply_));
    this->command_queue_.finish();
}


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::compute_reference_results(
    graphlily::aligned_dense_float_vec_t &mask,
    graphlily::aligned_dense_float_vec_t &inout,
    uint32_t len,
    float val
) {
    if (this->mask_type_ == graphlily::kMaskWriteToZero) {
        for (size_t i = 0; i < len; i++) {
            if (mask[i] == 0) {
                inout[i] = val;
            }
        }
    } else if (this->mask_type_ == graphlily::kMaskWriteToOne) {
        for (size_t i = 0; i < len; i++) {
            if (mask[i] != 0) {
                inout[i] = val;
            }
        }
    } else {
        std::cout << "Invalid mask type" << std::endl;
        exit(EXIT_FAILURE);
    }
}

}  // namespace module
}  // namespace graphlily

#endif  // GRAPHLILY_ASSIGN_VECTOR_DENSE_MODULE_H_
