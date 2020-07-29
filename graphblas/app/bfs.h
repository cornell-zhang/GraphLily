#ifndef __GRAPHBLAS_APP_BFS_H
#define __GRAPHBLAS_APP_BFS_H

#include "./module_collection.h"
#include "../module/spmv_module.h"
#include "../module/assign_vector_dense_module.h"

namespace graphblas {
namespace app {

class BFS : public app::ModuleCollection {
private:
    // modules
    using matrix_data_t = bool;
    using vector_data_t = unsigned int;
    graphblas::module::SpMVModule<matrix_data_t, vector_data_t> *SpMV_;
    graphblas::module::AssignVectorDenseModule<vector_data_t> *Assign_;
    // Sparse matrix size
    uint32_t matrix_num_rows_;
    uint32_t matrix_num_cols_;

public:
    BFS(std::string csr_float_npz_path) {
        this->SpMV_ = new graphblas::module::SpMVModule<matrix_data_t, vector_data_t>(csr_float_npz_path,
                                                                                    graphblas::kLogicalAndOr,
                                                                                    2,
                                                                                    10000);
        this->SpMV_->set_mask_type(graphblas::kMaskWriteToZero);
        this->matrix_num_rows_ = this->SpMV_->get_num_rows();
        this->matrix_num_cols_ = this->SpMV_->get_num_cols();
        this->add_module(this->SpMV_);
        this->Assign_ = new graphblas::module::AssignVectorDenseModule<vector_data_t>();
        this->Assign_->set_mask_type(graphblas::kMaskWriteToOne);
        this->add_module(this->Assign_);
    }

    using aligned_vector_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;
    aligned_vector_t run(uint32_t source, uint32_t num_iterations) {
        aligned_vector_t input(this->matrix_num_rows_, 0);
        aligned_vector_t level(this->matrix_num_rows_, 0);
        input[source] = 1;
        level[source] = 1;
        for (size_t i = 1; i <= num_iterations; i++) {
            input = this->SpMV_->run(input, level);
            this->Assign_->run(input, level, this->matrix_num_rows_, i+1);
        }
        return level;
    }

    graphblas::aligned_float_t compute_reference_results(uint32_t source, uint32_t num_iterations) {
        graphblas::aligned_float_t input(this->matrix_num_rows_, 0);
        graphblas::aligned_float_t level(this->matrix_num_rows_, 0);
        input[source] = 1;
        level[source] = 1;
        for (size_t i = 1; i <= num_iterations; i++) {
            input = this->SpMV_->compute_reference_results(input, level);
            this->Assign_->compute_reference_results(input, level, this->matrix_num_rows_, i+1);
        }
        return level;
    }
};

}  // namespace app
}  // namespace graphblas

#endif // __GRAPHBLAS_APP_BFS_H