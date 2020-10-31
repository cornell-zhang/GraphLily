#ifndef __GRAPHBLAS_EWISE_ADD_MODULE_H
#define __GRAPHBLAS_EWISE_ADD_MODULE_H

#include <cstdint>
#include <vector>
#include <fstream>
#include <chrono>

#include "xcl2.hpp"

#include "../global.h"
#include "./base_module.h"


namespace graphblas {
namespace module {

template<typename vector_data_t>
class eWiseAddModule : public BaseModule {
private:
    using packed_val_t = struct {vector_data_t data[graphblas::pack_size];};
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
    eWiseAddModule() : BaseModule("kernel_add_scalar_vector_dense") {}

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
        this->kernel_.setArg(0, this->in_buf);
    }

    /*!
     * \brief Bind the output buffer to an existing buffer.
     */
    void bind_out_buf(cl::Buffer src_buf) {
        this->out_buf = src_buf;
        this->kernel_.setArg(1, this->out_buf);
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
    graphblas::aligned_dense_float_vec_t
    compute_reference_results(graphblas::aligned_dense_float_vec_t const &in,
                              uint32_t len,
                              float val);

    void generate_kernel_header() override;

    void generate_kernel_ini() override;
};


template<typename vector_data_t>
void eWiseAddModule<vector_data_t>::generate_kernel_header() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream header(graphblas::proj_folder_name + "/" + this->kernel_name_ + ".h");
    // Data types
    header << "const unsigned PACK_SIZE = " << graphblas::pack_size << ";" << std::endl;
    header << "typedef struct {VAL_T data[PACK_SIZE];}" << " PACKED_VAL_T;" << std::endl;
    header.close();
}


template<typename vector_data_t>
void eWiseAddModule<vector_data_t>::generate_kernel_ini() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream ini(graphblas::proj_folder_name + "/" + this->kernel_name_ + ".ini");
    ini << "[connectivity]" << std::endl;
    ini << "sp=kernel_add_scalar_vector_dense_1.in:DDR[0]" << std::endl;
    ini << "sp=kernel_add_scalar_vector_dense_1.out:DDR[0]" << std::endl;
    ini.close();
}


template<typename vector_data_t>
void eWiseAddModule<vector_data_t>::send_in_host_to_device(aligned_dense_vec_t &in) {
    this->in_.assign(in.begin(), in.end());
    cl_mem_ext_ptr_t in_ext;
    in_ext.obj = this->in_.data();
    in_ext.param = 0;
    in_ext.flags = graphblas::DDR[0];
    cl_int err;
    OCL_CHECK(err, this->in_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * this->in_.size(),
                &in_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(0, this->in_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->in_buf}, 0));
    this->command_queue_.finish();
}


template<typename vector_data_t>
void eWiseAddModule<vector_data_t>::allocate_out_buf(uint32_t len) {
    this->out_.resize(len);
    cl_mem_ext_ptr_t out_ext;
    out_ext.obj = this->out_.data();
    out_ext.param = 0;
    out_ext.flags = graphblas::DDR[0];
    cl_int err;
    OCL_CHECK(err, this->out_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * this->out_.size(),
                &out_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(1, this->out_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->out_buf}, 0));
    this->command_queue_.finish();
}


template<typename vector_data_t>
void eWiseAddModule<vector_data_t>::run(uint32_t len, vector_data_t val) {
    cl_int err;
    OCL_CHECK(err, err = this->kernel_.setArg(2, len));
    // To avoid runtime error of invalid scalar argument size
    if (std::is_same<vector_data_t, ap_ufixed<32, 1>>::value) {
        OCL_CHECK(err, err = this->kernel_.setArg(3, 8, (void*)&val));
    } else {
        OCL_CHECK(err, err = this->kernel_.setArg(3, val));
    }
    OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->kernel_));
    this->command_queue_.finish();
}


template<typename vector_data_t> graphblas::aligned_dense_float_vec_t
eWiseAddModule<vector_data_t>::compute_reference_results(graphblas::aligned_dense_float_vec_t const &in,
                                                         uint32_t len,
                                                         float val) {
    graphblas::aligned_dense_float_vec_t out(len);
    for (uint32_t i = 0; i < len; i++) {
        out[i] = in[i] + val;
    }
    return out;
}


} // namespace module
} // namespace graphblas

#endif // __GRAPHBLAS_EWISE_ADD_MODULE_H
