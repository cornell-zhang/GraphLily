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

    /*! \brief Generate new frontier (used in SSSP) or not (used in BFS) */
    bool generate_new_frontier_;
    /*! \brief Internal copy of mask */
    aligned_mask_t mask_;
    /*! \brief Internal copy of inout */
    aligned_dense_vec_t inout_;
    /*! \brief Internal copy of new_frontier */
    aligned_mask_t new_frontier_;

public:
    // Device buffers
    cl::Buffer mask_buf;
    cl::Buffer inout_buf;
    cl::Buffer new_frontier_buf;

public:
    AssignVectorSparseModule(bool generate_new_frontier) : BaseModule("overlay") {
        this->generate_new_frontier_ = generate_new_frontier;
    }

    /* SpMSpV apply overlay argument list:
    * Index       Argument                              Used in this module?
    * 0           vector for spmv                       n
    * 1           mask for spmv (read port)             n
    * 2           mask for spmv (write port)            n
    * 3           output for spmv                       n
    *
    * 4~6         matrix for spmspv                     n
    * 7           vector for spmspv                     y
    * 8           mask for spmspv                       y
    * 9           output for spmspv                     y
    *
    * 10          number of rows                        n
    * 11          number of columns                     n
    * 12          semiring operation type               n
    *
    * 13          mask type                             n
    * 14          overlay mode select                   y
    * 15          apply vector length                   n
    * 16          apply input value or semiring zero    y
    */
    void set_unused_args() override {
        // Set unused arguments for SpMV and SpMSpV
        for (size_t i = 0; i <= 6; ++i) {
            this->spmspv_apply_.setArg(i, cl::Buffer(this->context_, 0, 4));
        }
        // Set unused scalar arguments
        this->spmspv_apply_.setArg(10, (unsigned)NULL);
        this->spmspv_apply_.setArg(11, (unsigned)NULL);
        this->spmspv_apply_.setArg(12, (char)NULL);
        this->spmspv_apply_.setArg(13, (char)NULL);
        this->spmspv_apply_.setArg(15, (unsigned)NULL);
        // Set unused arguments depending on `generate_new_frontier`
        if (!this->generate_new_frontier_) {
            // no new frontier, and no need for spmspv output
            this->spmspv_apply_.setArg(9, cl::Buffer(this->context_, 0, 4));
        } else {
            // generate new frontier, and no need for input value
            this->spmspv_apply_.setArg(16, (unsigned)NULL);
        }
    }

    void set_mode() override {
        if (this->generate_new_frontier_) {
            this->spmspv_apply_.setArg(14, 6);
        } else {
            this->spmspv_apply_.setArg(14, 5);
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
        if (this->generate_new_frontier_) {
            this->spmspv_apply_.setArg(9, this->mask_buf);
        } else {
            this->spmspv_apply_.setArg(7, this->mask_buf);
        }
    }

    /*!
     * \brief Bind the inout buffer to an existing buffer.
     */
    void bind_inout_buf(cl::Buffer src_buf) {
        this->inout_buf = src_buf;
        this->spmspv_apply_.setArg(8, this->inout_buf);
    }

    /*!
     * \brief Bind the new_frontier buffer to an existing buffer.
     */
    void bind_new_frontier_buf(cl::Buffer src_buf) {
        if (!this->generate_new_frontier_) {
            std::cout << "[ERROR]: this->generate_new_frontier_ should be true" << std::endl;
            exit(EXIT_FAILURE);
        }
        this->new_frontier_buf = src_buf;
        this->spmspv_apply_.setArg(7, this->new_frontier_buf);
    }

    /*!
     * \brief Run the module when this->generate_new_frontier_ is false (BFS mode).
     * \param val The value to be assigned to the inout vector.
     */
    void run(vector_data_t val);

    /*!
     * \brief Run the module when this->generate_new_frontier_ is true (SSSP mode).
     */
    void run();

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
        if (!this->generate_new_frontier_) {
            std::cout << "[ERROR]: this->generate_new_frontier_ should be true" << std::endl;
            exit(EXIT_FAILURE);
        }
        this->command_queue_.enqueueMigrateMemObjects({this->new_frontier_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->command_queue_.finish();
        return this->new_frontier_;
    }

    /*!
     * \brief Compute reference results when this->generate_new_frontier_ is false (BFS mode).
     * \param mask The mask vector.
     * \param inout The inout vector.
     * \param val The value to be assigned to the inout vector.
     */
    void compute_reference_results(graphlily::aligned_sparse_float_vec_t &mask,
                                   graphlily::aligned_dense_float_vec_t &inout,
                                   float val);

    /*!
     * \brief Compute reference results when this->generate_new_frontier_ is true (SSSP mode).
     * \param mask The mask vector.
     * \param inout The inout vector.
     * \param new_frontier The new frontier.
     */
    void compute_reference_results(graphlily::aligned_sparse_float_vec_t &mask,
                                   graphlily::aligned_dense_float_vec_t &inout,
                                   graphlily::aligned_sparse_float_vec_t &new_frontier);
};


template<typename vector_data_t, typename sparse_vector_data_t>
void AssignVectorSparseModule<vector_data_t, sparse_vector_data_t>::send_mask_host_to_device(
    aligned_mask_t &mask
) {
    cl_int err;
    // handle mask
    this->mask_.assign(mask.begin(), mask.end());
    cl_mem_ext_ptr_t mask_ext;
    mask_ext.obj = this->mask_.data();
    mask_ext.param = 0;
    if (this->generate_new_frontier_) {
        mask_ext.flags = graphlily::HBM[22];
    } else {
        mask_ext.flags = graphlily::HBM[20];
    }
    OCL_CHECK(err, this->mask_buf = cl::Buffer(this->context_,
                CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                sizeof(sparse_vector_data_t) * this->mask_.size(),
                &mask_ext,
                &err));
    if (this->generate_new_frontier_) {
        this->spmspv_apply_.setArg(9, this->mask_buf);
    } else {
        this->spmspv_apply_.setArg(7, this->mask_buf);
    }
    if (this->generate_new_frontier_) {
        // allocate memory for new_frontier
        this->new_frontier_.resize(this->mask_.size());
        cl_mem_ext_ptr_t new_frontier_ext;
        new_frontier_ext.obj = this->new_frontier_.data();
        new_frontier_ext.param = 0;
        new_frontier_ext.flags = graphlily::HBM[20];
        OCL_CHECK(err, this->new_frontier_buf = cl::Buffer(this->context_,
                    CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                    sizeof(sparse_vector_data_t) * this->new_frontier_.size(),
                    &new_frontier_ext,
                    &err));
        OCL_CHECK(err, err = this->spmspv_apply_.setArg(7, this->new_frontier_buf));
        OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->new_frontier_buf}, 0));
        this->command_queue_.finish();
    }
}


template<typename vector_data_t, typename sparse_vector_data_t>
void AssignVectorSparseModule<vector_data_t, sparse_vector_data_t>::send_inout_host_to_device(
    aligned_dense_vec_t &inout
) {
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
    OCL_CHECK(err, err = this->spmspv_apply_.setArg(8, this->inout_buf));
    OCL_CHECK(err, err = this->command_queue_.enqueueMigrateMemObjects({this->inout_buf}, 0));
    this->command_queue_.finish();
}


template<typename vector_data_t, typename sparse_vector_data_t>
void AssignVectorSparseModule<vector_data_t, sparse_vector_data_t>::run(vector_data_t val) {
    if (this->generate_new_frontier_) {
        std::cout << "[ERROR]: this->generate_new_frontier_ should be false" << std::endl;
        exit(EXIT_FAILURE);
    }
    cl_int err;
    OCL_CHECK(err, err = this->spmspv_apply_.setArg(16, graphlily::pack_raw_bits_to_uint(val)));

    this->command_queue_.enqueueTask(this->spmspv_apply_);
    this->command_queue_.finish();
}


template<typename vector_data_t, typename sparse_vector_data_t>
void AssignVectorSparseModule<vector_data_t, sparse_vector_data_t>::run() {
    if (!this->generate_new_frontier_) {
        std::cout << "[ERROR]: this->generate_new_frontier_ should be true" << std::endl;
        exit(EXIT_FAILURE);
    }
    this->command_queue_.enqueueTask(this->spmspv_apply_);
    this->command_queue_.finish();
}


template<typename vector_data_t, typename sparse_vector_data_t>
void AssignVectorSparseModule<vector_data_t, sparse_vector_data_t>::compute_reference_results(
    graphlily::aligned_sparse_float_vec_t &mask,
    graphlily::aligned_dense_float_vec_t &inout,
    float val
) {
    for (size_t i = 0; i < mask[0].index; i++) {
        inout[mask[i + 1].index] = val;
    }
}


template<typename vector_data_t, typename sparse_vector_data_t>
void AssignVectorSparseModule<vector_data_t, sparse_vector_data_t>::compute_reference_results(
    graphlily::aligned_sparse_float_vec_t &mask,
    graphlily::aligned_dense_float_vec_t &inout,
    graphlily::aligned_sparse_float_vec_t &new_frontier
) {
    new_frontier.clear();
    for (size_t i = 0; i < mask[0].index; i++) {
        if (inout[mask[i + 1].index] > mask[i + 1].val) {
            inout[mask[i + 1].index] = mask[i + 1].val;
            new_frontier.push_back(mask[i + 1]);
        }
    }
    graphlily::idx_float_t new_frontier_head;
    new_frontier_head.index = new_frontier.size();
    new_frontier_head.val = 0;
    new_frontier.insert(new_frontier.begin(), new_frontier_head);
}

}  // namespace module
}  // namespace graphlily

#endif  // GRAPHLILY_ASSIGN_VECTOR_SPARSE_MODULE_H_
