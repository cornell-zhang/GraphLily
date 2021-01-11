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

public:
    // Device buffers
    cl::Buffer in_buf;
    cl::Buffer out_buf;

public:
    eWiseAddModule() : BaseModule("overlay") {}

    /*Overlay argument list:
    * (H = num_hbm_channels)
    * Index       Argument                     used in this module?
    * 0 ~ H-1     matrix for spmv              n
    * H+0         vector for spmv              y
    * H+1         mask for spmv (read port)    n
    * H+2         mask for spmv (write port)   n
    * H+3         output for spmv              y
    *
    * H+4 ~ +6    matrix for spmspv            n
    * H+7         vector for spmspv            n
    * H+8         mask for spmspv              n
    * H+9         output for spmspv            n
    *
    * H+10        # of rows                    n
    * H+11        # of columns                 n
    *
    * H+12        operation type               n
    * H+13        mask type                    n
    *
    * H+14        overlay mode select          y
    *
    * H+15        apply vector length          y
    * H+16        input value for assign       y
    */
    void set_unused_args() override {
        // Set unused arguments for SpMV
        for (uint32_t i = 0; i < graphlily::num_hbm_channels; i++) {
            this->kernel_.setArg(i, cl::Buffer(this->context_, 0, 4));
        }
        this->kernel_.setArg(graphlily::num_hbm_channels + 1, cl::Buffer(this->context_, 0, 4));
        this->kernel_.setArg(graphlily::num_hbm_channels + 2, cl::Buffer(this->context_, 0, 4));
        // Set unused arguments for SpMSpV
        for (uint32_t i = graphlily::num_hbm_channels + 4; i <= graphlily::num_hbm_channels + 9; i++) {
            this->kernel_.setArg(i, cl::Buffer(this->context_, 0, 4));
        }
        // Set unused scalar arguments
        this->kernel_.setArg(graphlily::num_hbm_channels + 10, (unsigned)NULL);
        this->kernel_.setArg(graphlily::num_hbm_channels + 11, (unsigned)NULL);
        this->kernel_.setArg(graphlily::num_hbm_channels + 12, (char)NULL);
        this->kernel_.setArg(graphlily::num_hbm_channels + 13, (char)NULL);
    }

    void set_mode() override {
        this->kernel_.setArg(graphlily::num_hbm_channels + 14, 3);  // 3 is kernel_add_scalar_vector_dense
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
        this->kernel_.setArg(graphlily::num_hbm_channels + 3, this->in_buf);
    }

    /*!
     * \brief Bind the output buffer to an existing buffer.
     */
    void bind_out_buf(cl::Buffer src_buf) {
        this->out_buf = src_buf;
        this->kernel_.setArg(graphlily::num_hbm_channels + 0, this->out_buf);
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
    OCL_CHECK(err, err = this->kernel_.setArg(graphlily::num_hbm_channels + 3, this->in_buf));
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
    OCL_CHECK(err, err = this->kernel_.setArg(graphlily::num_hbm_channels + 0, this->out_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->out_buf}, 0));
    this->command_queue_.finish();
}


template<typename vector_data_t>
void eWiseAddModule<vector_data_t>::run(uint32_t len, vector_data_t val) {
    cl_int err;
    // TODO: is the overhead of setArg and enqueueTask large at run time?
    OCL_CHECK(err, err = this->kernel_.setArg(graphlily::num_hbm_channels + 15, len));
    // To avoid runtime error of invalid scalar argument size
    if (!(std::is_same<vector_data_t, unsigned>::value || std::is_same<vector_data_t, float>::value)) {
        OCL_CHECK(err, err = this->kernel_.setArg(graphlily::num_hbm_channels + 16, 8, (void*)&val));
    } else {
        OCL_CHECK(err, err = this->kernel_.setArg(graphlily::num_hbm_channels + 16, val));
    }
    OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->kernel_));
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
