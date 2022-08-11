#ifndef GRAPHLILY_EWISE_ADD_MODULE_H_
#define GRAPHLILY_EWISE_ADD_MODULE_H_

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
class eWiseAddModule : public BaseModule {
private:
    using packed_val_t = struct {vector_data_t data[graphlily::pack_size];};
    using aligned_dense_vec_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;

    /*! \brief Internal copy of the input vector */
    aligned_dense_vec_t in_;
    /*! \brief Internal copy of the output vector */
    aligned_dense_vec_t out_;
    /*! \brief The offset of spmv(VL, RD), spmspv, and apply kernel
               equals to 4 + spmspv HBM channels */
    uint32_t overlay_arg_offset_;

public:
    // Device buffers
    cl::Buffer in_buf;
    cl::Buffer out_buf;

public:
    eWiseAddModule() : BaseModule() {
        this->overlay_arg_offset_ = 4 + graphlily::spmspv_num_hbm_channels;
    }

    /* Overlay (VL, RD of SpMV, SpMSpV, and apply) argument list:
    *  (k = 4 + 1 * SpMSpV HBM channels)
    * Index       Argument                              Used in this module?
    * 0           vector for spmv                       y
    * 1           mask for spmv (read port)             n
    * 2           mask for spmv (write port)            n
    * 3           output for spmv                       y
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
    * k+6         mask type                             n
    * k+7         overlay mode select                   y
    * k+8         apply vector length                   y
    * k+9         apply input value or semiring zero    y
    * k+10        spmv RD row partition index           n
    */
    void set_unused_args() override {
        // Set unused arguments for SpMV
        this->spmv_vl_rd_spmspv_apply_.setArg(1, cl::Buffer(this->context_, 0, 4));
        this->spmv_vl_rd_spmspv_apply_.setArg(2, cl::Buffer(this->context_, 0, 4));
        // Set unused arguments for SpMSpV
        for (size_t i = 4; i <= this->overlay_arg_offset_ + 2; ++i) {
            this->spmv_vl_rd_spmspv_apply_.setArg(i, cl::Buffer(this->context_, 0, 4));
        }
        // Set unused scalar arguments
        this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 3, (unsigned)NULL);
        this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 4, (unsigned)NULL);
        this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 5, (char)NULL);
        this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 6, (char)NULL);
        this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 10, (unsigned)NULL);
    }

    void set_mode() override {
        this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 7, 3);  // 3 is kernel_add_scalar_vector_dense
    }

    /*!
     * \brief Send the input vector from host to device.
     */
    void send_in_host_to_device(aligned_dense_vec_t &in);

    /*!
     * \brief Allocate the output buffer.
     */
    void allocate_out_buf(uint32_t len);

    /*!
     * \brief Bind the input buffer to an existing buffer.
     */
    void bind_in_buf(cl::Buffer src_buf) {
        this->in_buf = src_buf;
        this->spmv_vl_rd_spmspv_apply_.setArg(3, this->in_buf);
    }

    /*!
     * \brief Bind the output buffer to an existing buffer.
     */
    void bind_out_buf(cl::Buffer src_buf) {
        this->out_buf = src_buf;
        this->spmv_vl_rd_spmspv_apply_.setArg(0, this->out_buf);
    }

    /*!
     * \brief Run the module.
     * \param len The length of the in/out vector.
     * \param val The value to be added.
     */
    void run(uint32_t len, vector_data_t val);

    /*!
     * \brief Send the output vector from device to host.
     * \return The output vector.
     */
    aligned_dense_vec_t send_out_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->out_;
    }

    /*!
     * \brief Compute reference results.
     * \param in The inout vector.
     * \param len The length of the mask/inout vector.
     * \param val The value to be assigned to the inout vector.
     * \return The output vector.
     */
    graphlily::aligned_dense_float_vec_t
    compute_reference_results(graphlily::aligned_dense_float_vec_t const &in,
                              uint32_t len,
                              float val);
};


template<typename vector_data_t>
void eWiseAddModule<vector_data_t>::send_in_host_to_device(aligned_dense_vec_t &in) {
    this->in_.assign(in.begin(), in.end());
    cl_mem_ext_ptr_t in_ext;
    in_ext.obj = this->in_.data();
    in_ext.param = 0;
    in_ext.flags = graphlily::HBM[22];
    cl_int err;
    OCL_CHECK(err, this->in_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * this->in_.size(),
                &in_ext,
                &err));
    OCL_CHECK(err, err = this->spmv_vl_rd_spmspv_apply_.setArg(3, this->in_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->in_buf}, 0));
    this->command_queue_.finish();
}


template<typename vector_data_t>
void eWiseAddModule<vector_data_t>::allocate_out_buf(uint32_t len) {
    this->out_.resize(len);
    cl_mem_ext_ptr_t out_ext;
    out_ext.obj = this->out_.data();
    out_ext.param = 0;
    out_ext.flags = graphlily::HBM[20];
    cl_int err;
    OCL_CHECK(err, this->out_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * this->out_.size(),
                &out_ext,
                &err));
    OCL_CHECK(err, err = this->spmv_vl_rd_spmspv_apply_.setArg(0, this->out_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->out_buf}, 0));
    this->command_queue_.finish();
}


template<typename vector_data_t>
void eWiseAddModule<vector_data_t>::run(uint32_t len, vector_data_t val) {
    cl_int err;
    // TODO: is the overhead of setArg and enqueueTask large at run time?
    OCL_CHECK(err, err = this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 8, len));
    OCL_CHECK(err, err = this->spmv_vl_rd_spmspv_apply_.setArg(this->overlay_arg_offset_ + 9, graphlily::pack_raw_bits_to_uint(val)));

    OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->spmv_vl_rd_spmspv_apply_));
    this->command_queue_.finish();
}


template<typename vector_data_t> graphlily::aligned_dense_float_vec_t
eWiseAddModule<vector_data_t>::compute_reference_results(graphlily::aligned_dense_float_vec_t const &in,
                                                         uint32_t len,
                                                         float val) {
    graphlily::aligned_dense_float_vec_t out(len);
    for (uint32_t i = 0; i < len; i++) {
        out[i] = in[i] + val;
    }
    return out;
}

}  // namespace module
}  // namespace graphlily

#endif  // GRAPHLILY_EWISE_ADD_MODULE_H_
