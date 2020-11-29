#ifndef GRAPHLILY_APP_BFS_H_
#define GRAPHLILY_APP_BFS_H_

#include "graphlily/app/module_collection.h"
#include "graphlily/module/spmv_module.h"
#include "graphlily/module/spmspv_module.h"
#include "graphlily/module/assign_vector_dense_module.h"
#include "graphlily/module/assign_vector_sparse_module.h"
#include "graphlily/io/data_loader.h"
#include "graphlily/io/data_formatter.h"


namespace graphlily {
namespace app {

class BFS : public app::ModuleCollection {
private:
    // modules
    module::SpMVModule<graphlily::val_t, graphlily::val_t> *SpMV_;
    module::AssignVectorDenseModule<graphlily::val_t> *DenseAssign_;
    module::SpMSpVModule<graphlily::val_t, graphlily::val_t, graphlily::idx_val_t> *SpMSpV_;
    module::AssignVectorSparseModule<graphlily::val_t, graphlily::idx_val_t> *SparseAssign_;
    // Sparse matrix size
    uint32_t matrix_num_rows_;
    uint32_t matrix_num_cols_;
    // SpMV kernel configuration
    uint32_t num_channels_;
    uint32_t out_buf_len_;
    uint32_t vec_buf_len_;

    using aligned_dense_vec_t = graphlily::aligned_dense_vec_t;
    using aligned_sparse_vec_t = graphlily::aligned_sparse_vec_t;
    using aligned_dense_float_vec_t = graphlily::aligned_dense_float_vec_t;

public:
    BFS(uint32_t num_channels, uint32_t out_buf_len, uint32_t vec_buf_len) {
        this->num_channels_ = num_channels;
        this->out_buf_len_ = out_buf_len;
        this->vec_buf_len_ = vec_buf_len;

        this->SpMV_ = new module::SpMVModule<graphlily::val_t, graphlily::val_t>(
            this->num_channels_,
            this->out_buf_len_,
            this->vec_buf_len_);
        this->SpMV_->set_semiring(graphlily::LogicalSemiring);
        this->SpMV_->set_mask_type(graphlily::kMaskWriteToZero);
        this->add_module(this->SpMV_);

        this->DenseAssign_ = new module::AssignVectorDenseModule<graphlily::val_t>();
        this->DenseAssign_->set_mask_type(graphlily::kMaskWriteToOne);
        this->add_module(this->DenseAssign_);

        this->SpMSpV_ = new module::SpMSpVModule<graphlily::val_t, graphlily::val_t, graphlily::idx_val_t>(
            out_buf_len);
        this->SpMSpV_->set_semiring(graphlily::LogicalSemiring);
        this->SpMSpV_->set_mask_type(graphlily::kMaskWriteToZero);
        this->add_module(this->SpMSpV_);

        bool generate_new_frontier = false;
        this->SparseAssign_ = new module::AssignVectorSparseModule<graphlily::val_t, graphlily::idx_val_t>(
            generate_new_frontier);
        this->add_module(this->SparseAssign_);
    }


    uint32_t get_nnz() {
        return this->SpMV_->get_nnz();
    }


    void load_and_format_matrix(std::string csr_float_npz_path, bool skip_empty_rows) {
        CSRMatrix<float> csr_matrix = graphlily::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
        graphlily::io::util_round_csr_matrix_dim(
            csr_matrix,
            this->num_channels_ * graphlily::pack_size * graphlily::num_cycles_float_add,
            this->num_channels_ * graphlily::pack_size * graphlily::num_cycles_float_add);
        for (auto &x : csr_matrix.adj_data) x = 1;
        CSCMatrix<float> csc_matrix = graphlily::io::csr2csc(csr_matrix);
        this->SpMV_->load_and_format_matrix(csr_matrix, skip_empty_rows);
        this->SpMSpV_->load_and_format_matrix(csc_matrix);
        this->matrix_num_rows_ = this->SpMV_->get_num_rows();
        this->matrix_num_cols_ = this->SpMV_->get_num_cols();
        assert(this->matrix_num_rows_ == this->matrix_num_cols_);
    }


    void send_matrix_host_to_device() {
        this->SpMV_->send_matrix_host_to_device();
        this->SpMSpV_->send_matrix_host_to_device();
    }


    aligned_dense_vec_t pull(uint32_t source, uint32_t num_iterations) {
        aligned_dense_vec_t input(this->matrix_num_rows_, 0);
        aligned_dense_vec_t distance(this->matrix_num_rows_, 0);
        input[source] = 1;
        distance[source] = 1;
        this->SpMV_->send_vector_host_to_device(input);
        this->SpMV_->send_mask_host_to_device(distance);
        this->DenseAssign_->bind_mask_buf(this->SpMV_->vector_buf);
        this->DenseAssign_->bind_inout_buf(this->SpMV_->mask_buf);
        for (size_t i = 1; i <= num_iterations; i++) {
            this->SpMV_->run();
            this->SpMV_->copy_buffer_device_to_device(this->SpMV_->results_buf,
                                                      this->SpMV_->vector_buf,
                                                      sizeof(graphlily::val_t) * this->matrix_num_rows_);
            this->DenseAssign_->run(this->matrix_num_rows_, i + 1);
        }
        return this->SpMV_->send_mask_device_to_host();
    }


    aligned_dense_vec_t push(uint32_t source, uint32_t num_iterations) {
        // The sparse input vector
        aligned_sparse_vec_t spmspv_input(2);
        idx_val_t head;
        graphlily::idx_t nnz = 1;  // one source vertex
        head.index = nnz;
        spmspv_input[0] = head;
        spmspv_input[1] = {source, 1};

        // The dense distance vector
        aligned_dense_vec_t distance(this->matrix_num_rows_, 0);
        distance[source] = 1;

        // Push
        this->SpMSpV_->send_vector_host_to_device(spmspv_input);
        this->SpMSpV_->send_mask_host_to_device(distance);
        this->SparseAssign_->bind_mask_buf(this->SpMSpV_->vector_buf);
        this->SparseAssign_->bind_inout_buf(this->SpMSpV_->mask_buf);
        for (size_t i = 1; i <= num_iterations; i++) {
            this->SpMSpV_->run();
            // TODO: the size of buffer to be copied should be sizeof(graphlily::idx_val_t) * (1 + nnz)
            this->SpMSpV_->copy_buffer_device_to_device(
                this->SpMSpV_->results_buf,
                this->SpMSpV_->vector_buf,
                sizeof(graphlily::idx_val_t) * (1 + this->matrix_num_rows_));
            this->SparseAssign_->run(i + 1);
        }

        return this->SpMSpV_->send_mask_device_to_host();
    }


    // aligned_dense_vec_t pull_push(uint32_t source, uint32_t num_iterations) {
    //     // The sparse input vector
    //     aligned_sparse_vec_t spmspv_input(2);
    //     idx_val_t head;
    //     graphlily::idx_t nnz = 1; // one source vertex
    //     head.index = nnz;
    //     spmspv_input[0] = head;
    //     spmspv_input[1] = {source, 1};

    //     // The dense distance vector
    //     aligned_dense_vec_t distance(this->matrix_num_rows_, 0);
    //     distance[source] = 1;

    //     // Push
    //     this->SpMSpV_->send_vector_host_to_device(spmspv_input);
    //     this->SpMSpV_->send_mask_host_to_device(distance);
    //     this->SparseAssign_->bind_mask_buf(this->SpMSpV_->vector_buf);
    //     this->SparseAssign_->bind_inout_buf(this->SpMSpV_->mask_buf);
    //     for (size_t i = 1; i < 5; i++) {
    //         this->SpMSpV_->run();
    //         this->SpMSpV_->copy_buffer_device_to_device(
    //             this->SpMSpV_->results_buf,
    //             this->SpMSpV_->vector_buf,
    //             sizeof(graphlily::idx_val_t) * (1 + this->matrix_num_rows_));
    //         this->SparseAssign_->run(i + 1);
    //     }

    //     // Switch
    //     this->SpMV_->send_mask_host_to_device(distance);  // TODO: SpMV and SpMSpV should share distance
    //     this->SpMSpV_->copy_buffer_device_to_device(this->SpMSpV_->mask_buf,
    //                                                 this->SpMV_->mask_buf,
    //                                                 sizeof(graphlily::val_t) * this->matrix_num_rows_);
    //     aligned_sparse_vec_t spmspv_result = this->SpMSpV_->send_results_device_to_host();
    //     aligned_dense_vec_t spmv_input = graphlily::convert_sparse_vec_to_dense_vec<aligned_sparse_vec_t,
    //         aligned_dense_vec_t>(spmspv_result, this->matrix_num_rows_);

    //     // Pull
    //     this->SpMV_->send_vector_host_to_device(spmv_input);
    //     this->DenseAssign_->bind_mask_buf(this->SpMV_->vector_buf);
    //     this->DenseAssign_->bind_inout_buf(this->SpMV_->mask_buf);
    //     for (size_t i = 5; i <= num_iterations; i++) {
    //         this->SpMV_->run();
    //         this->SpMV_->copy_buffer_device_to_device(
    //             this->SpMV_->results_buf,
    //             this->SpMV_->vector_buf,
    //             sizeof(graphlily::val_t) * this->matrix_num_rows_);
    //         this->DenseAssign_->run(this->matrix_num_rows_, i + 1);
    //     }

    //     return this->SpMV_->send_mask_device_to_host();
    // }


    aligned_dense_float_vec_t compute_reference_results(uint32_t source, uint32_t num_iterations) {
        aligned_dense_float_vec_t input(this->matrix_num_rows_, 0);
        aligned_dense_float_vec_t distance(this->matrix_num_rows_, 0);
        input[source] = 1;
        distance[source] = 1;
        for (size_t i = 1; i <= num_iterations; i++) {
            input = this->SpMV_->compute_reference_results(input, distance);
            this->DenseAssign_->compute_reference_results(input, distance, this->matrix_num_rows_, i + 1);
        }
        return distance;
    }
};

}  // namespace app
}  // namespace graphlily

#endif  // GRAPHLILY_APP_BFS_H_
