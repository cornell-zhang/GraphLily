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
    /*! \brief The mask type */
    graphblas::MaskType mask_type_;
    /*! \brief The packed data type */
    using packed_data_t = struct {vector_data_t data[graphblas::pack_size];};
    /*! \brief String representation of the data type */
    std::string vector_data_t_str_;

    // Device buffers
    cl::Buffer mask_buf_;
    cl::Buffer inout_buf_;

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

    void generate_kernel_header() override;

    void send_data_to_FPGA() override {} // Does nothing

    using aligned_vector_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;
    /*!
     * \brief Run the module.
     * \param mask The mask vector.
     * \param inout The inout vector.
     * \param length The length of the mask/inout vector.
     * \param val The value to be assigned to the inout vector.
     */
    void run(aligned_vector_t &mask, aligned_vector_t &inout,
             uint32_t length, vector_data_t val);

    /*!
     * \brief Compute reference results.
     * \param mask The mask vector.
     * \param inout The inout vector.
     * \param length The length of the mask/inout vector.
     * \param val The value to be assigned to the inout vector.
     */
    void compute_reference_results(graphblas::aligned_float_t &mask, graphblas::aligned_float_t &inout,
                                   uint32_t length, float val);
};


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::generate_kernel_header() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream header(graphblas::proj_folder_name + "/" + this->kernel_name_ + ".h");
    // Data types
    header << "typedef " << this->vector_data_t_str_ << " VECTOR_T;" << std::endl;
    header << "const unsigned int VECTOR_PACK_SIZE = " << graphblas::pack_size << ";" << std::endl;
    header << "typedef struct {VECTOR_T data[VECTOR_PACK_SIZE];}" << " PACKED_VECTOR_T;" << std::endl;
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
void AssignVectorDenseModule<vector_data_t>::run(aligned_vector_t &mask,
                                                 aligned_vector_t &inout,
                                                 uint32_t length,
                                                 vector_data_t val) {
    cl_int err;
    cl_mem_ext_ptr_t mask_ext;
    mask_ext.obj = mask.data();
    mask_ext.param = 0;
    mask_ext.flags = 0;
    OCL_CHECK(err, this->mask_buf_ = cl::Buffer(this->context_,
                CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * length,
                &mask_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(0, this->mask_buf_));
    cl_mem_ext_ptr_t inout_ext;
    inout_ext.obj = inout.data();
    inout_ext.param = 0;
    inout_ext.flags = 0;
    OCL_CHECK(err, this->inout_buf_ = cl::Buffer(this->context_,
                CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(vector_data_t) * length,
                &inout_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(1, this->inout_buf_));
    OCL_CHECK(err, err = this->kernel_.setArg(2, length));
    OCL_CHECK(err, err = this->kernel_.setArg(3, val));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->mask_buf_, this->inout_buf_}, 0));
    OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->kernel_));
    this->command_queue_.finish();
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->inout_buf_},
        CL_MIGRATE_MEM_OBJECT_HOST));
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
