#ifndef __GRAPHBLAS_ASSIGN_VECTOR_DENSE_MODULE_H
#define __GRAPHBLAS_ASSIGN_VECTOR_DENSE_MODULE_H

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
class AssignVectorDenseModule : public BaseModule {
private:
    using packed_val_t = struct {vector_data_t data[graphblas::pack_size];};
    using aligned_val_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;

    /*! \brief The mask type */
    graphblas::MaskType mask_type_;
    /*! \brief String representation of the data type */
    std::string vector_data_t_str_;
    /*! \brief Internal copy of mask */
    aligned_val_t mask_;
    /*! \brief Internal copy of inout */
    aligned_val_t inout_;

public:
    // Device buffers
    cl::Buffer mask_buf;
    cl::Buffer inout_buf;

public:
    AssignVectorDenseModule() : BaseModule("kernel_assign_vector_dense") {
        this->vector_data_t_str_ = graphblas::dtype_to_str<vector_data_t>();
    }

    /*!
     * \brief Set the mask type.
     * \param mask_type The mask type.
     */
    void set_mask_type(graphblas::MaskType mask_type) {
        if (mask_type == graphblas::kNoMask) {
            std::cerr << "Please set the mask type" << std::endl;
            exit(EXIT_FAILURE);
        } else {
            this->mask_type_ = mask_type;
        }
    }

    /*!
     * \brief Send the mask from host to device.
     */
    void send_mask_host_to_device(aligned_val_t &mask);

    /*!
     * \brief Send the inout from host to device.
     */
    void send_inout_host_to_device(aligned_val_t &inout);

    /*!
     * \brief Bind the mask buffer to an existing buffer.
     */
    void bind_mask_buf(cl::Buffer src_buf) {
        this->mask_buf = src_buf;
        this->kernel_.setArg(0, this->mask_buf);
    }

    /*!
     * \brief Bind the inout buffer to an existing buffer.
     */
    void bind_inout_buf(cl::Buffer src_buf) {
        this->inout_buf = src_buf;
        this->kernel_.setArg(1, this->inout_buf);
    }

    /*!
     * \brief Run the module.
     * \param length The length of the mask/inout vector.
     * \param val The value to be assigned to the inout vector.
     */
    void run(uint32_t length, vector_data_t val);

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
     * \brief Send the inout from device to host.
     * \return The inout.
     */
    aligned_val_t send_inout_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->inout_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->inout_;
    }

    /*!
     * \brief Compute reference results.
     * \param mask The mask vector.
     * \param inout The inout vector.
     * \param length The length of the mask/inout vector.
     * \param val The value to be assigned to the inout vector.
     */
    void compute_reference_results(graphblas::aligned_float_t &mask,
                                   graphblas::aligned_float_t &inout,
                                   uint32_t length,
                                   float val);

    void generate_kernel_header() override;

    void generate_kernel_ini() override;
};


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::generate_kernel_header() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream header(graphblas::proj_folder_name + "/" + this->kernel_name_ + ".h");
    // Data types
    header << "typedef " << this->vector_data_t_str_ << " VAL_T;" << std::endl;
    header << "const unsigned int PACK_SIZE = " << graphblas::pack_size << ";" << std::endl;
    header << "typedef struct {VAL_T data[PACK_SIZE];}" << " PACKED_VAL_T;" << std::endl;
    // Mask
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
    header.close();
}


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::generate_kernel_ini() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream ini(graphblas::proj_folder_name + "/" + this->kernel_name_ + ".ini");
    ini << "[connectivity]" << std::endl;
    ini << "sp=kernel_assign_vector_dense_1.mask:DDR[0]" << std::endl;
    ini << "sp=kernel_assign_vector_dense_1.inout:DDR[0]" << std::endl;
    ini.close();
}


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::send_mask_host_to_device(aligned_val_t &mask) {
    this->mask_.assign(mask.begin(), mask.end());
    cl_mem_ext_ptr_t mask_ext;
    mask_ext.obj = this->mask_.data();
    mask_ext.param = 0;
    mask_ext.flags = graphblas::DDR[0];
    cl_int err;
    OCL_CHECK(err, this->mask_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * this->mask_.size(),
                &mask_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(0, this->mask_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->mask_buf}, 0));
    this->command_queue_.finish();
}


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::send_inout_host_to_device(aligned_val_t &inout) {
    this->inout_.assign(inout.begin(), inout.end());
    cl_mem_ext_ptr_t inout_ext;
    inout_ext.obj = this->inout_.data();
    inout_ext.param = 0;
    inout_ext.flags = graphblas::DDR[0];
    cl_int err;
    OCL_CHECK(err, this->inout_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * this->inout_.size(),
                &inout_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(1, this->inout_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->inout_buf}, 0));
    this->command_queue_.finish();
}


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::run(uint32_t length, vector_data_t val) {
    cl_int err;
    OCL_CHECK(err, err = this->kernel_.setArg(2, length));
    // To avoid runtime error of invalid scalar argument size
    if (std::is_same<vector_data_t, ap_ufixed<32, 1>>::value) {
        OCL_CHECK(err, err = this->kernel_.setArg(3, 8, (void*)&val));
    } else {
        OCL_CHECK(err, err = this->kernel_.setArg(3, val));
    }
    OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->kernel_));
    this->command_queue_.finish();
}


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::compute_reference_results(graphblas::aligned_float_t &mask,
                                                                       graphblas::aligned_float_t &inout,
                                                                       uint32_t length,
                                                                       float val) {
    if (this->mask_type_ == graphblas::kMaskWriteToZero) {
        for (size_t i = 0; i < length; i++) {
            if (mask[i] == 0) {
                inout[i] = val;
            }
        }
    } else if (this->mask_type_ == graphblas::kMaskWriteToOne) {
        for (size_t i = 0; i < length; i++) {
            if (mask[i] != 0) {
                inout[i] = val;
            }
        }
    } else {
        std::cout << "Invalid mask type" << std::endl;
        exit(EXIT_FAILURE);
    }
}


} // namespace module
} // namespace graphblas

#endif // __GRAPHBLAS_ASSIGN_VECTOR_DENSE_MODULE_H
