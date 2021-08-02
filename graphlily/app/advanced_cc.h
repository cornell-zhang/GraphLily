#ifndef GRAPHLILY_APP_ADVANCEDCC_H_
#define GRAPHLILY_APP_ADVANCEDCC_H_

#include "graphlily/app/module_collection.h"
#include "graphlily/module/spmv_module.h"
#include "graphlily/module/spmspv_module.h"
#include "graphlily/module/assign_vector_dense_module.h"
#include "graphlily/module/assign_vector_sparse_module.h"
#include "graphlily/module/add_scalar_vector_dense_module.h"
#include "graphlily/io/data_loader.h"
#include "graphlily/io/data_formatter.h"

#include <iostream>
#include <chrono>

namespace graphlily {
namespace app {
class AdvancedCC : public app::ModuleCollection 
{
private:
    // modules
    module::SpMVModule<graphlily::val_t, graphlily::val_t> *SpMV_;
    module::AssignVectorDenseModule<graphlily::val_t> *DenseAssign_;
    module::SpMSpVModule<graphlily::val_t, graphlily::val_t, graphlily::idx_val_t> *SpMSpV_;
    module::AssignVectorSparseModule<graphlily::val_t, graphlily::idx_val_t> *SparseAssign_;
    module::eWiseAddModule<graphlily::val_t> *eWiseAdd_;  // for on-device data transfer
    // Sparse matrix size
    uint32_t matrix_num_rows_;
    uint32_t matrix_num_cols_;
    // SpMV kernel configuration
    uint32_t num_channels_;
    uint32_t spmv_out_buf_len_;
    uint32_t spmspv_out_buf_len_;
    uint32_t vec_buf_len_;
    // Semiring
    graphlily::SemiringType semiring_ = graphlily::MinSelSecSemiring;
    // Data types
    using aligned_dense_vec_t = graphlily::aligned_dense_vec_t;
    using aligned_sparse_vec_t = graphlily::aligned_sparse_vec_t;
    using aligned_dense_float_vec_t = graphlily::aligned_dense_float_vec_t;
public:
    AdvancedCC(uint32_t num_channels, uint32_t spmv_out_buf_len,
            uint32_t spmspv_out_buf_len, uint32_t vec_buf_len){
        this->num_channels_ = num_channels;
        this->spmv_out_buf_len_ = spmv_out_buf_len;
        this->spmspv_out_buf_len_ = spmspv_out_buf_len;
        this->vec_buf_len_ = vec_buf_len;
        //SpMV Initialization
        this->SpMV_ = new module::SpMVModule<graphlily::val_t, graphlily::val_t>(
            this->num_channels_,
            this->spmv_out_buf_len_,
            this->vec_buf_len_);
        this->SpMV_->set_semiring(semiring_);
        this->SpMV_->set_mask_type(graphlily::kNoMask);
        this->add_module(this->SpMV_);
        //Dense assign
        this->DenseAssign_ = new module::AssignVectorDenseModule<graphlily::val_t>();
        this->DenseAssign_->set_mask_type(graphlily::kMaskWriteToOne);
        this->add_module(this->DenseAssign_);
        //SpMSpV initialization
        this->SpMSpV_ = new module::SpMSpVModule<graphlily::val_t, graphlily::val_t, graphlily::idx_val_t>(
            spmspv_out_buf_len);
        this->SpMSpV_->set_semiring(semiring_);
        this->SpMSpV_->set_mask_type(graphlily::kMaskWriteToZero);
        this->add_module(this->SpMSpV_);
        //SparseAssign
        bool generate_new_frontier = false;
        this->SparseAssign_ = new module::AssignVectorSparseModule<graphlily::val_t, graphlily::idx_val_t>(
            generate_new_frontier);
        this->add_module(this->SparseAssign_);

        this->eWiseAdd_ = new module::eWiseAddModule<graphlily::val_t>();
        this->add_module(this->eWiseAdd_);
    }
    void preprocess(CSRMatrix<float>& csr_matrix) {
        // add self edges and set their weights to one
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
                csr_matrix.adj_data.insert(csr_matrix.adj_data.begin() + start, float(1));
                csr_matrix.adj_indices.insert(csr_matrix.adj_indices.begin() + start, row_idx);
                nnz_each_row[row_idx]++;
            } else {
                bool add_self_edge = false;
                for (size_t i = start; i < end; i++) {
                    uint32_t col_idx = csr_matrix.adj_indices[i];
                    if (col_idx == row_idx) {
                        csr_matrix.adj_data[i] = float(1);
                        break;
                    } else if (col_idx > row_idx) {
                        add_self_edge = true;
                        csr_matrix.adj_data.insert(csr_matrix.adj_data.begin() + i, float(1));
                        csr_matrix.adj_indices.insert(csr_matrix.adj_indices.begin() + i, row_idx);
                        break;
                    } else if (i == (end - 1)) {
                        add_self_edge = true;
                        csr_matrix.adj_data.insert(csr_matrix.adj_data.begin() + i, float(1));
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
    uint32_t get_nnz() {
        return this->SpMV_->get_nnz();
    }
    void load_and_format_matrix(std::string csr_float_npz_path, bool skip_empty_rows) {
        CSRMatrix<float> csr_matrix = graphlily::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
        this->preprocess(csr_matrix);
        graphlily::io::util_round_csr_matrix_dim(
            csr_matrix,
            this->num_channels_ * graphlily::pack_size,
            this->num_channels_ * graphlily::pack_size);
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
    aligned_dense_vec_t pull_mult(){
        aligned_dense_vec_t label_vec(this->matrix_num_rows_, 0);
        for (auto i = 0; i < this->matrix_num_rows_; i++)
        {
            label_vec[i] = i + 1;
        }
        aligned_dense_vec_t tag(this->matrix_num_rows_, 0);
        this->SpMV_->send_vector_host_to_device(label_vec);
        this->SpMV_->send_mask_host_to_device(tag);
        this->eWiseAdd_->bind_in_buf(this->SpMV_->results_buf);
        this->eWiseAdd_->bind_out_buf(this->SpMV_->vector_buf);
        this->SpMV_->run();
        this->eWiseAdd_->run(this->matrix_num_rows_, 0);
        return this->SpMV_->send_vector_device_to_host();
    }
    aligned_dense_vec_t pull() {
        aligned_dense_vec_t label_vec(this->matrix_num_rows_, 0);
        for (auto i = 0; i < this->matrix_num_rows_; i++)
        {
            label_vec[i] = i + 1;
        }
        aligned_dense_vec_t tag(this->matrix_num_rows_, 0);
        aligned_dense_vec_t label_vec_buf(this->matrix_num_rows_, 0);
        this->SpMV_->send_vector_host_to_device(label_vec);
        this->SpMV_->send_mask_host_to_device(tag);
        this->eWiseAdd_->bind_in_buf(this->SpMV_->results_buf);
        this->eWiseAdd_->bind_out_buf(this->SpMV_->vector_buf);
        while(true){
            this->SpMV_->run();
            this->eWiseAdd_->run(this->matrix_num_rows_, 0);
            label_vec = this->SpMV_->send_vector_device_to_host();
            if (compare_vector_on_kernel(label_vec, label_vec_buf))
            {
                break;
            }
            memcpy(label_vec_buf, label_vec);
        }
        return this->SpMV_->send_vector_device_to_host();
    }
    bool compare_vector_on_kernel(aligned_dense_vec_t& a, aligned_dense_vec_t& b){
        bool ret = true;
        for(size_t i = 0; i < a.size(); i++){
            if(a[i] != b[i]){
                ret = false;
                break;
            }
        }
        return ret;
    }
    void memcpy(aligned_dense_vec_t& a, aligned_dense_vec_t& b){
        for(size_t i = 0; i < a.size(); i++){
            a[i] = b[i];
        }
    }
};
}
}
#endif