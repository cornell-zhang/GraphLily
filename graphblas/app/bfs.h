#ifndef __GRAPHBLAS_APP_BFS_H
#define __GRAPHBLAS_APP_BFS_H

#include "./module_collection.h"
#include "../module/spmv_module.h"
#include "../module/spmspv_module.h"
#include "../module/assign_vector_dense_module.h"
#include "../module/assign_vector_sparse_module.h"
#include "../io/data_loader.h"
#include "../io/data_formatter.h"


namespace graphblas {
namespace app {

class BFS : public app::ModuleCollection {
private:
    // modules
    using matrix_data_t = unsigned;
    using vector_data_t = unsigned;
    using index_val_t = struct {graphblas::idx_t index; vector_data_t val;};
    graphblas::module::SpMVModule<matrix_data_t, vector_data_t> *SpMV_;
    graphblas::module::AssignVectorDenseModule<vector_data_t> *DenseAssign_;
    graphblas::module::SpMSpVModule<matrix_data_t, vector_data_t, index_val_t> *SpMSpV_;
    graphblas::module::AssignVectorSparseModule<vector_data_t, index_val_t> *SparseAssign_;
    // Sparse matrix size
    uint32_t matrix_num_rows_;
    uint32_t matrix_num_cols_;
    // SpMV kernel configuration
    static const graphblas::SemiRingType semiring_ = graphblas::kLogicalAndOr;
    uint32_t num_channels_;
    uint32_t out_buf_len_;
    uint32_t vec_buf_len_;

public:
    BFS(uint32_t num_channels, uint32_t out_buf_len, uint32_t vec_buf_len) {
        this->num_channels_ = num_channels;
        this->out_buf_len_ = out_buf_len
        this->vec_buf_len_ = vec_buf_len;
        this->SpMV_ = new graphblas::module::SpMVModule<matrix_data_t, vector_data_t>(semiring_,
                                                                                      this->num_channels_,
                                                                                      this->out_buf_len_,
                                                                                      this->vec_buf_len_);
        this->SpMV_->set_mask_type(graphblas::kMaskWriteToZero);
        this->add_module(this->SpMV_);
        this->DenseAssign_ = new graphblas::module::AssignVectorDenseModule<vector_data_t>();
        this->DenseAssign_->set_mask_type(graphblas::kMaskWriteToOne);
        this->add_module(this->DenseAssign_);
        this->SpMSpV_ = new graphblas::module::SpMSpVModule<matrix_data_t, vector_data_t, index_val_t>(
            semiring_, out_buf_len);
        this->SpMSpV_->set_mask_type(graphblas::kMaskWriteToZero);
        this->add_module(this->SpMSpV_);
        this->SparseAssign_ = new graphblas::module::AssignVectorSparseModule<vector_data_t, index_val_t>();
        this->SparseAssign_->set_mode(0);
        this->add_module(this->SparseAssign_);
    }

    uint32_t get_nnz() {
        return this->SpMV_->get_nnz();
    }

    void load_and_format_matrix(std::string csr_float_npz_path) {
        CSRMatrix<float> csr_matrix = graphblas::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
        graphblas::io::util_round_csr_matrix_dim(csr_matrix,
                                                 num_channels_ * graphblas::pack_size,
                                                 num_channels_ * graphblas::pack_size);
        for (auto &x : csr_matrix.adj_data) x = 1;
        CSCMatrix<float> csc_matrix = graphblas::io::csr2csc(csr_matrix);
        this->SpMV_->load_and_format_matrix(csr_matrix);
        this->SpMSpV_->load_and_format_matrix(csc_matrix);
        this->matrix_num_rows_ = this->SpMV_->get_num_rows();
        this->matrix_num_cols_ = this->SpMV_->get_num_cols();
        assert(this->matrix_num_rows_ == this->matrix_num_cols_);
    }

    void send_matrix_host_to_device() {
        this->SpMV_->send_matrix_host_to_device();
        this->SpMSpV_->send_matrix_host_to_device();
    }

    using aligned_dense_vec_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;
    aligned_dense_vec_t run_pull_only(uint32_t source, uint32_t num_iterations) {
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
                                                      sizeof(vector_data_t) * this->matrix_num_rows_);
            this->DenseAssign_->run(this->matrix_num_rows_, i+1);
        }
        return this->SpMV_->send_mask_device_to_host();
    }


    using aligned_sparse_vec_t = std::vector<index_val_t, aligned_allocator<index_val_t>>;
    aligned_dense_vec_t run_push_only(uint32_t source, uint32_t num_iterations) {
        // The sparse input vector
        aligned_sparse_vec_t spmspv_input(2);
        index_val_t head;
        graphblas::idx_t nnz = 1; // one source vertex
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
        this->SparseAssign_->bind_new_frontier_buf(this->SpMSpV_->mask_buf); // TODO: This is dangerous
        for (size_t i = 1; i <= num_iterations; i++) {
            this->SpMSpV_->run();
            this->SpMSpV_->copy_buffer_device_to_device(this->SpMSpV_->results_buf,
                                                        this->SpMSpV_->vector_buf,
                                                        sizeof(index_val_t) * (1 + this->matrix_num_rows_));
            this->SparseAssign_->run(i + 1);
        }

        return this->SpMSpV_->send_mask_device_to_host();
    }


    aligned_dense_vec_t run_pull_push(uint32_t source, uint32_t num_iterations) {
        // The sparse input vector
        aligned_sparse_vec_t spmspv_input(2);
        index_val_t head;
        graphblas::idx_t nnz = 1; // one source vertex
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
        this->SparseAssign_->bind_new_frontier_buf(this->SpMSpV_->mask_buf); // TODO: This is dangerous
        for (size_t i = 1; i < 5; i++) {
            this->SpMSpV_->run();
            this->SpMSpV_->copy_buffer_device_to_device(this->SpMSpV_->results_buf,
                                                        this->SpMSpV_->vector_buf,
                                                        sizeof(index_val_t) * (1 + this->matrix_num_rows_));
            this->SparseAssign_->run(i + 1);
        }

        // Switch
        this->SpMV_->send_mask_host_to_device(distance); // TODO: SpMV and SpMSpV should share distance
        this->SpMSpV_->copy_buffer_device_to_device(this->SpMSpV_->mask_buf,
                                                    this->SpMV_->mask_buf,
                                                    sizeof(vector_data_t) * this->matrix_num_rows_);
        aligned_sparse_vec_t spmspv_result = this->SpMSpV_->send_results_device_to_host();
        aligned_dense_vec_t spmv_input =
            graphblas::convert_sparse_vec_to_dense_vec<aligned_sparse_vec_t, aligned_dense_vec_t>(spmspv_result,
                                                                                                  this->matrix_num_rows_);

        // Pull
        this->SpMV_->send_vector_host_to_device(spmv_input);
        this->DenseAssign_->bind_mask_buf(this->SpMV_->vector_buf);
        this->DenseAssign_->bind_inout_buf(this->SpMV_->mask_buf);
        for (size_t i = 5; i <= num_iterations; i++) {
            this->SpMV_->run();
            this->SpMV_->copy_buffer_device_to_device(this->SpMV_->results_buf,
                                                      this->SpMV_->vector_buf,
                                                      sizeof(vector_data_t) * this->matrix_num_rows_);
            this->DenseAssign_->run(this->matrix_num_rows_, i + 1);
        }

        return this->SpMV_->send_mask_device_to_host();
    }


    graphblas::aligned_dense_float_vec_t compute_reference_results(uint32_t source, uint32_t num_iterations) {
        graphblas::aligned_dense_float_vec_t input(this->matrix_num_rows_, 0);
        graphblas::aligned_dense_float_vec_t distance(this->matrix_num_rows_, 0);
        input[source] = 1;
        distance[source] = 1;
        for (size_t i = 1; i <= num_iterations; i++) {
            input = this->SpMV_->compute_reference_results(input, distance);
            this->DenseAssign_->compute_reference_results(input, distance, this->matrix_num_rows_, i+1);
        }
        return distance;
    }
};

}  // namespace app
}  // namespace graphblas

#endif // __GRAPHBLAS_APP_BFS_H
