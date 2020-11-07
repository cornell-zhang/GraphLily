#ifndef GRAPHLILY_ASSIGN_VECTOR_SPARSE_MODULE_H_
#define GRAPHLILY_ASSIGN_VECTOR_SPARSE_MODULE_H_

#include <cstdint>
#include <vector>
#include <fstream>
#include <chrono>

#include "xcl2.hpp"

#include "graphlily/global.h"
#include "graphlily/module/base_module.h"


namespace graphlily {
namespace module {

template<typename vector_data_t, typename sparse_vector_data_t>
class AssignVectorSparseModule : public BaseModule {
private:
    using aligned_mask_t = std::vector<sparse_vector_data_t, aligned_allocator<sparse_vector_data_t>>;
    using aligned_dense_vec_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;

    /*! \brief Internal copy of mask */
    aligned_mask_t mask_;
    /*! \brief Internal copy of inout */
    aligned_dense_vec_t inout_;
    /*! \brief Internal copy of new_frontier */
    aligned_mask_t new_frontier_;
    /*! \brief Working mode. 0 for BFS, 1 for SSSP */
    unsigned mode_;

public:
    // Device buffers
    cl::Buffer mask_buf;
    cl::Buffer inout_buf;
    cl::Buffer new_frontier_buf;

public:
    AssignVectorSparseModule() : BaseModule("kernel_assign_vector_sparse") {}

    /*!
     * \brief Set the working mode.
     * \param mode The working mode. 0 for BFS, 1 for SSSP.
     */
    void set_mode(unsigned mode) {
        if (mode > 1) {
            std::cerr << "Invalid mode configuration" << std::endl;
            exit(EXIT_FAILURE);
        } else {
            this->mode_ = mode;
        }
    }

    /*!
     * \brief Send the mask from host to device.
     */
    void send_mask_host_to_device(aligned_mask_t &mask);

    /*!
     * \brief Send the inout from host to device.
     */
    void send_inout_host_to_device(aligned_dense_vec_t &inout);

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
     * \brief Bind the new_frontier buffer to an existing buffer.
     */
    void bind_new_frontier_buf(cl::Buffer src_buf) {
        this->new_frontier_buf = src_buf;
        this->kernel_.setArg(2, this->new_frontier_buf);
    }

    /*!
     * \brief Run the module.
     * \param len The length of the mask/inout vector.
     * \param val The value to be assigned to the inout vector.
     */
    void run(vector_data_t val);

    /*!
     * \brief Send the mask from device to host.
     * \return The mask.
     */
    aligned_mask_t send_mask_device_to_host() {
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
     * \brief Send the new_frontier from device to host.
     * \return The inout.
     */
    aligned_mask_t send_new_frontier_device_to_host() {
        this->command_queue_.enqueueMigrateMemObjects({this->new_frontier_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->new_frontier_;
    }

    /*!
     * \brief Compute reference results.
     * \param mask The mask vector.
     * \param inout The inout vector.
     * \param len The length of the mask/inout vector.
     * \param val The value to be assigned to the inout vector.
     */
    void compute_reference_results(graphlily::aligned_sparse_float_vec_t &mask,
                                   graphlily::aligned_dense_float_vec_t &inout,
                                   graphlily::aligned_sparse_float_vec_t &new_frontier,
                                   float val);

    void generate_kernel_header() override;

    void generate_kernel_ini() override;
};


template<typename vector_data_t, typename sparse_vector_data_t>
void AssignVectorSparseModule<vector_data_t,sparse_vector_data_t>::generate_kernel_header() {
    std::string command = "mkdir -p " + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream header(graphlily::proj_folder_name + "/" + this->kernel_name_ + ".h");
    // Data types
    header << "typedef unsigned IDX_T;" << std::endl;
    header << "const unsigned BATCH_SIZE = " << 128 << ";" << std::endl;
    header << "typedef struct {IDX_T index; VAL_T val;}" << " VI_T;" << std::endl;
    header.close();
}


template<typename vector_data_t, typename sparse_vector_data_t>
void AssignVectorSparseModule<vector_data_t,sparse_vector_data_t>::generate_kernel_ini() {
    std::string command = "mkdir -p " + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream ini(graphlily::proj_folder_name + "/" + this->kernel_name_ + ".ini");
    ini << "[connectivity]" << std::endl;
    ini << "sp=kernel_assign_vector_sparse_1.mask:DDR[0]" << std::endl;
    ini << "sp=kernel_assign_vector_sparse_1.inout:DDR[0]" << std::endl;
    ini << "sp=kernel_assign_vector_sparse_1.new_frontier:DDR[0]" << std::endl;
    ini.close();
}


template<typename vector_data_t, typename sparse_vector_data_t>
void AssignVectorSparseModule<vector_data_t,sparse_vector_data_t>::send_mask_host_to_device(aligned_mask_t &mask) {
    cl_int err;
    // handle mask
    this->mask_.assign(mask.begin(), mask.end());
    cl_mem_ext_ptr_t mask_ext;
    mask_ext.obj = this->mask_.data();
    mask_ext.param = 0;
    mask_ext.flags = graphlily::DDR[0];
    OCL_CHECK(err, this->mask_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(sparse_vector_data_t) * this->mask_.size(),
                &mask_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(0, this->mask_buf));

    // allocate memory for new_frontier
    this->new_frontier_.resize(this->mask_.size());
    cl_mem_ext_ptr_t new_frontier_ext;
    new_frontier_ext.obj = this->new_frontier_.data();
    new_frontier_ext.param = 0;
    new_frontier_ext.flags = graphlily::DDR[0];
    OCL_CHECK(err, this->new_frontier_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(sparse_vector_data_t) * this->new_frontier_.size(),
                &new_frontier_ext,
                &err));
    OCL_CHECK(err, err = this->kernel_.setArg(2, this->new_frontier_buf));

    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->new_frontier_buf}, 0));
    this->command_queue_.finish();
}


template<typename vector_data_t, typename sparse_vector_data_t>
void AssignVectorSparseModule<vector_data_t,sparse_vector_data_t>::send_inout_host_to_device(aligned_dense_vec_t &inout) {
    this->inout_.assign(inout.begin(), inout.end());
    cl_mem_ext_ptr_t inout_ext;
    inout_ext.obj = this->inout_.data();
    inout_ext.param = 0;
    inout_ext.flags = graphlily::DDR[0];
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


template<typename vector_data_t, typename sparse_vector_data_t>
void AssignVectorSparseModule<vector_data_t,sparse_vector_data_t>::run(vector_data_t val) {
    cl_int err;
    OCL_CHECK(err, err = this->kernel_.setArg(3, this->mode_));
    // To avoid runtime error of invalid scalar argument size
    if (std::is_same<vector_data_t, ap_ufixed<32, 1>>::value) {
        OCL_CHECK(err, err = this->kernel_.setArg(4, 8, (void*)&val));
    } else {
        OCL_CHECK(err, err = this->kernel_.setArg(4, val));
    }
    std::cout << "[INFO AssignSparseModule] Kernel Started" << std::endl << std::flush;
    OCL_CHECK(err, err = this->command_queue_.enqueueTask(this->kernel_));
    this->command_queue_.finish();
}


template<typename vector_data_t, typename sparse_vector_data_t>
void AssignVectorSparseModule<vector_data_t,sparse_vector_data_t>::compute_reference_results(
    graphlily::aligned_sparse_float_vec_t &mask,
    graphlily::aligned_dense_float_vec_t &inout,
    graphlily::aligned_sparse_float_vec_t &new_frontier,
    float val
) {
    new_frontier.clear();
    if (this->mode_ == 0) {
        for (size_t i = 0; i < mask[0].index; i++) {
            inout[mask[i + 1].index] = val;
        }
    } else if (this->mode_ == 1) {
        for (size_t i = 0; i < mask[0].index; i++) {
            if (inout[mask[i + 1].index] > mask[i + 1].val) {
                inout[mask[i + 1].index] = mask[i + 1].val;
                new_frontier.push_back(mask[i+1]);
            }
        }
        graphlily::index_float_t nf_head;
        nf_head.index = new_frontier.size();
        nf_head.val = 0;
        new_frontier.insert(new_frontier.begin(),nf_head);
    }
    else {
        std::cout << "Invalid working mode" << std::endl;
        exit(EXIT_FAILURE);
    }
}

}  // namespace module
}  // namespace graphlily

#endif  // GRAPHLILY_ASSIGN_VECTOR_SPARSE_MODULE_H_
