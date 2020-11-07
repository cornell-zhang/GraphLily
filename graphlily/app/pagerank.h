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
    using matrix_data_t = ap_ufixed<32, 1>;
    using vector_data_t = ap_ufixed<32, 1>;
    graphlily::module::SpMVModule<matrix_data_t, vector_data_t> *SpMV_;
    graphlily::module::eWiseAddModule<vector_data_t> *eWiseAdd_;
    // Sparse matrix size
    uint32_t matrix_num_rows_;
    uint32_t matrix_num_cols_;
    // SpMV kernel configuration
    static const graphlily::SemiRingType semiring_ = graphlily::ArithmeticSemiring;
    uint32_t num_channels_;
    uint32_t out_buf_len_;
    uint32_t vec_buf_len_;

public:
    PageRank(uint32_t num_channels, uint32_t out_buf_len, uint32_t vec_buf_len) {
        this->num_channels_ = num_channels;
        this->out_buf_len_ = out_buf_len
        this->vec_buf_len_ = vec_buf_len;
        this->SpMV_ = new graphlily::module::SpMVModule<matrix_data_t, vector_data_t>(semiring_,
                                                                                      this->num_channels_,
                                                                                      this->out_buf_len_,
                                                                                      this->vec_buf_len_);
        this->SpMV_->set_mask_type(graphlily::kNoMask);
        this->add_module(this->SpMV_);
        this->eWiseAdd_ = new graphlily::module::eWiseAddModule<vector_data_t>();
        this->add_module(this->eWiseAdd_);
    }

    uint32_t get_nnz() {
        return this->SpMV_->get_nnz();
    }

    void load_and_format_matrix(std::string csr_float_npz_path, float damping) {
        CSRMatrix<float> csr_matrix = graphlily::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
        graphlily::io::util_round_csr_matrix_dim(csr_matrix,
                                                 num_channels_ * graphlily::pack_size,
                                                 num_channels_ * graphlily::pack_size);
        graphlily::io::util_normalize_csr_matrix_by_outdegree(csr_matrix);
        for (auto &x : csr_matrix.adj_data) x = x * damping;
        this->SpMV_->load_and_format_matrix(csr_matrix);
        this->matrix_num_rows_ = this->SpMV_->get_num_rows();
        this->matrix_num_cols_ = this->SpMV_->get_num_cols();
        assert(this->matrix_num_rows_ == this->matrix_num_cols_);
    }

    void send_matrix_host_to_device() {
        this->SpMV_->send_matrix_host_to_device();
    }

    using aligned_dense_vec_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;
    aligned_dense_vec_t run(vector_data_t damping, uint32_t num_iterations) {
        aligned_dense_vec_t rank(this->matrix_num_rows_, 1.0 / this->matrix_num_rows_);
        this->SpMV_->send_vector_host_to_device(rank);
        this->eWiseAdd_->bind_in_buf(this->SpMV_->results_buf);
        this->eWiseAdd_->bind_out_buf(this->SpMV_->vector_buf);
        for (size_t i = 1; i <= num_iterations; i++) {
            this->SpMV_->run();
            this->eWiseAdd_->run(this->matrix_num_rows_, (1 - damping) / this->matrix_num_rows_);
        }
        return this->SpMV_->send_vector_device_to_host();
    }

    graphlily::aligned_dense_float_vec_t compute_reference_results(float damping, uint32_t num_iterations) {
        graphlily::aligned_dense_float_vec_t rank(this->matrix_num_rows_, 1.0 / this->matrix_num_rows_);
        for (size_t i = 1; i <= num_iterations; i++) {
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
