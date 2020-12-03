#ifndef GRAPHLILY_APP_SSSP_H_
#define GRAPHLILY_APP_SSSP_H_

#include "graphlily/app/module_collection.h"
#include "graphlily/module/spmv_module.h"
#include "graphlily/module/spmspv_module.h"
#include "graphlily/module/assign_vector_sparse_module.h"
#include "graphlily/io/data_loader.h"
#include "graphlily/io/data_formatter.h"


namespace {

void _preprocess(CSRMatrix<float>& csr_matrix) {
    // randomly initialize edge weights
    for (uint32_t i = 0; i < csr_matrix.adj_data.size(); i++) {
        // csr_matrix.adj_data[i] = i % 10 + 1;
        csr_matrix.adj_data[i] = 1;  // When all edge weights are 1, SSSP becomes BFS
    }
    // add self edges and set their weights to zero
    uint32_t num_rows = csr_matrix.adj_indptr.size() - 1;
    std::vector<uint32_t> nnz_each_row(num_rows);
    for (size_t i = 0; i < num_rows; i++) {
        nnz_each_row[i] = csr_matrix.adj_indptr[i + 1] - csr_matrix.adj_indptr[i];
    }
    csr_matrix.adj_data.reserve(csr_matrix.adj_data.size() + num_rows);
    csr_matrix.adj_indices.reserve(csr_matrix.adj_indices.size() + num_rows);
    for (size_t row_idx = 0; row_idx < num_rows; row_idx++) {
        uint32_t start = csr_matrix.adj_indptr[row_idx];
        uint32_t end = csr_matrix.adj_indptr[row_idx + 1];
        if (start == end) {
            csr_matrix.adj_data.insert(csr_matrix.adj_data.begin() + start, float(0));
            csr_matrix.adj_indices.insert(csr_matrix.adj_indices.begin() + start, row_idx);
            nnz_each_row[row_idx]++;
        } else {
            bool add_self_edge = false;
            for (size_t i = start; i < end; i++) {
                uint32_t col_idx = csr_matrix.adj_indices[i];
                if (col_idx == row_idx) {
                    csr_matrix.adj_data[i] = float(0);
                    break;
                } else if (col_idx > row_idx) {
                    add_self_edge = true;
                    csr_matrix.adj_data.insert(csr_matrix.adj_data.begin() + i, float(0));
                    csr_matrix.adj_indices.insert(csr_matrix.adj_indices.begin() + i, row_idx);
                    break;
                } else if (i == (end - 1)) {
                    add_self_edge = true;
                    csr_matrix.adj_data.insert(csr_matrix.adj_data.begin() + i, float(0));
                    csr_matrix.adj_indices.insert(csr_matrix.adj_indices.begin() + i, row_idx);
                    break;
                }
            }
            if (add_self_edge) {
                nnz_each_row[row_idx]++;
            }
        }
        csr_matrix.adj_indptr[row_idx + 1] = csr_matrix.adj_indptr[row_idx] + nnz_each_row[row_idx];
    }
}

}  // namespace


namespace graphlily {
namespace app {

class SSSP : public app::ModuleCollection {
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
    // Semiring
    graphlily::SemiringType semiring_ = graphlily::TropicalSemiring;
    // Data types
    using aligned_dense_vec_t = graphlily::aligned_dense_vec_t;
    using aligned_sparse_vec_t = graphlily::aligned_sparse_vec_t;
    using aligned_dense_float_vec_t = graphlily::aligned_dense_float_vec_t;

public:
    SSSP(uint32_t num_channels, uint32_t out_buf_len, uint32_t vec_buf_len) {
        this->num_channels_ = num_channels;
        this->out_buf_len_ = out_buf_len;
        this->vec_buf_len_ = vec_buf_len;

        this->SpMV_ = new module::SpMVModule<graphlily::val_t, graphlily::val_t>(
            this->num_channels_,
            this->out_buf_len_,
            this->vec_buf_len_);
        this->SpMV_->set_semiring(semiring_);
        this->SpMV_->set_mask_type(graphlily::kNoMask);
        this->add_module(this->SpMV_);

        this->SpMSpV_ = new module::SpMSpVModule<graphlily::val_t, graphlily::val_t, graphlily::idx_val_t>(
            out_buf_len);
        this->SpMSpV_->set_semiring(semiring_);
        this->SpMSpV_->set_mask_type(graphlily::kNoMask);
        this->add_module(this->SpMSpV_);

        bool generate_new_frontier = true;
        this->SparseAssign_ = new module::AssignVectorSparseModule<graphlily::val_t, graphlily::idx_val_t>(
            generate_new_frontier);
        this->add_module(this->SparseAssign_);
    }


    uint32_t get_nnz() {
        return this->SpMV_->get_nnz();
    }


    void load_and_format_matrix(std::string csr_float_npz_path, bool skip_empty_rows) {
        CSRMatrix<float> csr_matrix = graphlily::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
        _preprocess(csr_matrix);
        graphlily::io::util_round_csr_matrix_dim(
            csr_matrix,
            this->num_channels_ * graphlily::pack_size * graphlily::num_cycles_float_add,
            this->num_channels_ * graphlily::pack_size * graphlily::num_cycles_float_add);
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
        aligned_dense_vec_t input(this->matrix_num_rows_, semiring_.zero);
        input[source] = 0;
        this->SpMV_->send_vector_host_to_device(input);
        for (size_t iter = 1; iter <= num_iterations; iter++) {
            this->SpMV_->run();
            this->SpMV_->copy_buffer_device_to_device(this->SpMV_->results_buf,
                                                      this->SpMV_->vector_buf,
                                                      sizeof(graphlily::val_t) * this->matrix_num_rows_);
        }
        return this->SpMV_->send_vector_device_to_host();
    }


    aligned_dense_vec_t push(uint32_t source, uint32_t num_iterations) {
        // The sparse input vector
        aligned_sparse_vec_t spmspv_input(2);
        idx_val_t head;
        graphlily::idx_t nnz = 1;  // one source vertex
        head.index = nnz;
        spmspv_input[0] = head;
        spmspv_input[1] = {source, 0};

        // The dense distance vector
        aligned_dense_vec_t distance(this->matrix_num_rows_, semiring_.zero);
        distance[source] = 0;

        // Push
        this->SpMSpV_->send_vector_host_to_device(spmspv_input);
        this->SpMSpV_->send_mask_host_to_device(distance);
        this->SparseAssign_->bind_mask_buf(this->SpMSpV_->results_buf);
        this->SparseAssign_->bind_inout_buf(this->SpMSpV_->mask_buf);
        this->SparseAssign_->bind_new_frontier_buf(this->SpMSpV_->vector_buf);
        for (size_t iter = 1; iter <= num_iterations; iter++) {
            this->SpMSpV_->run();
            this->SparseAssign_->run();
        }

        return this->SpMSpV_->send_mask_device_to_host();
    }


    aligned_dense_vec_t pull_push(uint32_t source, uint32_t num_iterations, float threshold = 0.05) {
        // The sparse input vector
        aligned_sparse_vec_t spmspv_input(2);
        idx_val_t head;
        graphlily::idx_t nnz = 1;  // one source vertex
        head.index = nnz;
        spmspv_input[0] = head;
        spmspv_input[1] = {source, 0};

        // The dense distance vector
        aligned_dense_vec_t distance(this->matrix_num_rows_, semiring_.zero);
        distance[source] = 0;

        // Push
        this->SpMSpV_->send_vector_host_to_device(spmspv_input);
        this->SpMSpV_->send_mask_host_to_device(distance);
        this->SparseAssign_->bind_mask_buf(this->SpMSpV_->results_buf);
        this->SparseAssign_->bind_inout_buf(this->SpMSpV_->mask_buf);
        this->SparseAssign_->bind_new_frontier_buf(this->SpMSpV_->vector_buf);
        uint32_t iter = 1;
        uint32_t vector_nnz;
        do {
            this->SpMSpV_->run();
            this->SparseAssign_->run();
            vector_nnz = this->SpMSpV_->get_results_nnz();
            iter++;
        } while (iter < num_iterations && (float(vector_nnz) / this->matrix_num_rows_ < threshold));

        // Switch from push to pull
        aligned_dense_vec_t spmv_input = this->SpMSpV_->send_mask_device_to_host();
        this->SpMV_->send_vector_host_to_device(spmv_input);

        // Pull
        for ( ; iter <= num_iterations; iter++) {
            this->SpMV_->run();
            this->SpMV_->copy_buffer_device_to_device(this->SpMV_->results_buf,
                                                      this->SpMV_->vector_buf,
                                                      sizeof(graphlily::val_t) * this->matrix_num_rows_);
        }

        return this->SpMV_->send_vector_device_to_host();
    }


    aligned_dense_float_vec_t compute_reference_results(uint32_t source, uint32_t num_iterations) {
        aligned_dense_float_vec_t input(this->matrix_num_rows_, semiring_.zero);
        input[source] = 0;
        for (size_t iter = 1; iter <= num_iterations; iter++) {
            input = this->SpMV_->compute_reference_results(input);
        }
        return input;
    }
};

}  // namespace app
}  // namespace graphlily

#endif  // GRAPHLILY_APP_SSSP_H_
