#ifndef __GRAPHBLAS_APP_BFS_H
#define __GRAPHBLAS_APP_BFS_H

#include "./module_collection.h"
#include "../module/spmv_module.h"
#include "../module/assign_vector_dense_module.h"
#include "../io/data_loader.h"
#include "../io/data_formatter.h"


namespace graphblas {
namespace app {

class BFS : public app::ModuleCollection {
private:
    // modules
    using matrix_data_t = unsigned int;
    using vector_data_t = unsigned int;
    graphblas::module::SpMVModule<matrix_data_t, vector_data_t> *SpMV_;
    graphblas::module::AssignVectorDenseModule<vector_data_t> *Assign_;
    // Sparse matrix size
    uint32_t matrix_num_rows_;
    uint32_t matrix_num_cols_;
    // SpMV kernel configuration
    static const graphblas::SemiRingType semiring_ = graphblas::kLogicalAndOr;
    uint32_t num_channels_;
    uint32_t out_buffer_len_;
    uint32_t vector_buffer_len_;

public:
    BFS(uint32_t num_channels, uint32_t out_buffer_len, uint32_t vector_buffer_len) {
        this->num_channels_ = num_channels;
        this->out_buffer_len_ = out_buffer_len;
        this->vector_buffer_len_ = vector_buffer_len;
        this->SpMV_ = new graphblas::module::SpMVModule<matrix_data_t, vector_data_t>(semiring_,
                                                                                      this->num_channels_,
                                                                                      this->out_buffer_len_,
                                                                                      this->vector_buffer_len_);
        this->SpMV_->set_mask_type(graphblas::kMaskWriteToZero);
        this->add_module(this->SpMV_);
        this->Assign_ = new graphblas::module::AssignVectorDenseModule<vector_data_t>();
        this->Assign_->set_mask_type(graphblas::kMaskWriteToOne);
        this->add_module(this->Assign_);
    }

    uint32_t get_nnz() {
        return this->SpMV_->get_nnz();
    }

    void load_and_format_matrix(std::string csr_float_npz_path) {
        struct CSRMatrix<float> csr_matrix = graphblas::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
        graphblas::io::util_round_csr_matrix_dim(csr_matrix,
                                                 num_channels_ * graphblas::pack_size,
                                                 num_channels_ * graphblas::pack_size);
        for (auto &x : csr_matrix.adj_data) x = 1;
        this->SpMV_->load_and_format_matrix(csr_matrix);
        this->matrix_num_rows_ = this->SpMV_->get_num_rows();
        this->matrix_num_cols_ = this->SpMV_->get_num_cols();
        assert(this->matrix_num_rows_ == this->matrix_num_cols_);
    }

    void send_matrix_host_to_device() {
        this->SpMV_->send_matrix_host_to_device();
    }

    using aligned_val_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;
    aligned_val_t run(uint32_t source, uint32_t num_iterations) {
        aligned_val_t input(this->matrix_num_rows_, 0);
        aligned_val_t distance(this->matrix_num_rows_, 0);
        input[source] = 1;
        distance[source] = 1;
        this->SpMV_->send_vector_host_to_device(input);
        this->SpMV_->send_mask_host_to_device(distance);
        this->Assign_->bind_mask_buf(this->SpMV_->vector_buf);
        this->Assign_->bind_inout_buf(this->SpMV_->mask_buf);
        for (size_t i = 1; i <= num_iterations; i++) {
            this->SpMV_->run();
            this->SpMV_->copy_buffer_device_to_device(this->SpMV_->results_buf,
                                                      this->SpMV_->vector_buf,
                                                      sizeof(vector_data_t) * this->matrix_num_rows_);
            this->Assign_->run(this->matrix_num_rows_, i+1);
        }
        return this->SpMV_->send_mask_device_to_host();
    }

    graphblas::aligned_float_t compute_reference_results(uint32_t source, uint32_t num_iterations) {
        graphblas::aligned_float_t input(this->matrix_num_rows_, 0);
        graphblas::aligned_float_t distance(this->matrix_num_rows_, 0);
        input[source] = 1;
        distance[source] = 1;
        for (size_t i = 1; i <= num_iterations; i++) {
            input = this->SpMV_->compute_reference_results(input, distance);
            this->Assign_->compute_reference_results(input, distance, this->matrix_num_rows_, i+1);
        }
        return distance;
    }
};

}  // namespace app
}  // namespace graphblas

#endif // __GRAPHBLAS_APP_BFS_H
