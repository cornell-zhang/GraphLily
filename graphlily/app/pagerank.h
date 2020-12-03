#ifndef GRAPHLILY_APP_PAGERANK_H_
#define GRAPHLILY_APP_PAGERANK_H_

#include "graphlily/app/module_collection.h"
#include "graphlily/module/spmv_module.h"
#include "graphlily/module/add_scalar_vector_dense_module.h"
#include "graphlily/io/data_loader.h"
#include "graphlily/io/data_formatter.h"


namespace graphlily {
namespace app {

class PageRank : public app::ModuleCollection {
private:
    // modules
    graphlily::module::SpMVModule<graphlily::val_t, graphlily::val_t> *SpMV_;
    graphlily::module::eWiseAddModule<graphlily::val_t> *eWiseAdd_;
    // Sparse matrix size
    uint32_t matrix_num_rows_;
    uint32_t matrix_num_cols_;
    // SpMV kernel configuration
    uint32_t num_channels_;
    uint32_t out_buf_len_;
    uint32_t vec_buf_len_;
    // Semiring
    graphlily::SemiringType semiring_ = graphlily::ArithmeticSemiring;
    // Data types
    using aligned_dense_vec_t = graphlily::aligned_dense_vec_t;
    using aligned_sparse_vec_t = graphlily::aligned_sparse_vec_t;
    using aligned_dense_float_vec_t = graphlily::aligned_dense_float_vec_t;

public:
    PageRank(uint32_t num_channels, uint32_t out_buf_len, uint32_t vec_buf_len) {
        this->num_channels_ = num_channels;
        this->out_buf_len_ = out_buf_len;
        this->vec_buf_len_ = vec_buf_len;

        this->SpMV_ = new graphlily::module::SpMVModule<graphlily::val_t, graphlily::val_t>(
            this->num_channels_,
            this->out_buf_len_,
            this->vec_buf_len_);
        this->SpMV_->set_semiring(semiring_);
        this->SpMV_->set_mask_type(graphlily::kNoMask);
        this->add_module(this->SpMV_);

        this->eWiseAdd_ = new graphlily::module::eWiseAddModule<graphlily::val_t>();
        this->add_module(this->eWiseAdd_);
    }


    uint32_t get_nnz() {
        return this->SpMV_->get_nnz();
    }


    void load_and_format_matrix(std::string csr_float_npz_path, float damping, bool skip_empty_rows) {
        CSRMatrix<float> csr_matrix = graphlily::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
        graphlily::io::util_round_csr_matrix_dim(
            csr_matrix,
            this->num_channels_ * graphlily::pack_size * graphlily::num_cycles_float_add,
            this->num_channels_ * graphlily::pack_size * graphlily::num_cycles_float_add);
        graphlily::io::util_normalize_csr_matrix_by_outdegree(csr_matrix);
        for (auto &x : csr_matrix.adj_data) x = x * damping;
        this->SpMV_->load_and_format_matrix(csr_matrix, skip_empty_rows);
        this->matrix_num_rows_ = this->SpMV_->get_num_rows();
        this->matrix_num_cols_ = this->SpMV_->get_num_cols();
        assert(this->matrix_num_rows_ == this->matrix_num_cols_);
    }


    void send_matrix_host_to_device() {
        this->SpMV_->send_matrix_host_to_device();
    }


    aligned_dense_vec_t pull(graphlily::val_t damping, uint32_t num_iterations) {
        aligned_dense_vec_t rank(this->matrix_num_rows_, 1.0 / this->matrix_num_rows_);
        this->SpMV_->send_vector_host_to_device(rank);
        this->eWiseAdd_->bind_in_buf(this->SpMV_->results_buf);
        this->eWiseAdd_->bind_out_buf(this->SpMV_->vector_buf);
        for (size_t iter = 1; iter <= num_iterations; iter++) {
            this->SpMV_->run();
            this->eWiseAdd_->run(this->matrix_num_rows_, (1 - damping) / this->matrix_num_rows_);
        }
        return this->SpMV_->send_vector_device_to_host();
    }


    aligned_dense_float_vec_t compute_reference_results(float damping, uint32_t num_iterations) {
        aligned_dense_float_vec_t rank(this->matrix_num_rows_, 1.0 / this->matrix_num_rows_);
        for (size_t iter = 1; iter <= num_iterations; iter++) {
            rank = this->SpMV_->compute_reference_results(rank);
            rank = this->eWiseAdd_->compute_reference_results(rank,
                                                              this->matrix_num_rows_,
                                                              (1 - damping) / this->matrix_num_rows_);
        }
        return rank;
    }
};

}  // namespace app
}  // namespace graphlily

#endif  // GRAPHLILY_APP_PAGERANK_H_
