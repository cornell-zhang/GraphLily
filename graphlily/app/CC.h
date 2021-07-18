#ifndef GRAPHLILY_APP_CC_H_
#define GRAPHLILY_APP_CC_H_

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
class CC : public app::ModuleCollection 
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
    graphlily::SemiringType semiring_ = graphlily::LogicalSemiring;
    // Data types
    using aligned_dense_vec_t = graphlily::aligned_dense_vec_t;
    using aligned_sparse_vec_t = graphlily::aligned_sparse_vec_t;
    using aligned_dense_float_vec_t = graphlily::aligned_dense_float_vec_t;
public:
    CC(uint32_t num_channels, uint32_t spmv_out_buf_len,
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
        this->SpMV_->set_mask_type(graphlily::kMaskWriteToZero);
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

    uint32_t get_nnz() {
        return this->SpMV_->get_nnz();
    }
    void load_and_format_matrix(std::string csr_float_npz_path, bool skip_empty_rows) {
        CSRMatrix<float> csr_matrix = graphlily::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
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
        std::cout << "rows: " << this->matrix_num_rows_ << std::endl;
    }
    void send_matrix_host_to_device() {
        this->SpMV_->send_matrix_host_to_device();
        this->SpMSpV_->send_matrix_host_to_device();
    }
    #define BFS_based_CC_by_pull 0
    #define BFS_based_CC_by_push 1
    #define BFS_based_CC_by_pull_push 2
    aligned_dense_vec_t BFS_based_CC(short option, uint32_t num_iterations, uint32_t num_components, float threshold = 0.05){
        aligned_dense_vec_t tot_tag(this->matrix_num_rows_, 0);
        int current_source = 0;
        for(size_t i = 1; i <= num_components; i++){
            aligned_dense_vec_t tag(this->matrix_num_rows_, 0);
            if (option == BFS_based_CC_by_pull)
            {
                tag = this->pull(current_source, num_iterations);
            }
            else if (option == BFS_based_CC_by_push){
                tag = this->push(current_source, num_iterations);
            }
            else{
                tag = this->pull_push(current_source, num_iterations, threshold);
            }
            for (size_t iter = 0; iter < tot_tag.size(); iter++){
                tot_tag[iter] = tot_tag[iter] + i * tag[iter];
            }
            current_source = this->calculate_source_for_kernel(tot_tag, current_source);
            if (current_source == -1)
            {
                break;
            }
        }
        return tot_tag;
    }
    aligned_dense_vec_t Fast_CC_pull(uint32_t num_iterations, uint32_t num_components){
        aligned_dense_vec_t tot_tag(this->matrix_num_rows_, 0);
        int current_source = 0;
        for(size_t i = 1; i <= num_components; i++){
            aligned_dense_vec_t tag_iter(this->matrix_num_rows_, 0);
            aligned_dense_vec_t input(this->matrix_num_rows_, semiring_.zero);
            aligned_dense_vec_t tag(this->matrix_num_rows_, 0);
            if (current_source == -1)
            {
                break;
            }
            input[current_source] = 1;
            tag[current_source] = 1;
            int mark = i;
            this->SpMV_->send_vector_host_to_device(input);
            this->SpMV_->send_mask_host_to_device(tag);
            this->DenseAssign_->bind_mask_buf(this->SpMV_->vector_buf);
            this->DenseAssign_->bind_inout_buf(this->SpMV_->mask_buf);
            this->eWiseAdd_->bind_in_buf(this->SpMV_->results_buf);
            this->eWiseAdd_->bind_out_buf(this->SpMV_->vector_buf);
            for (size_t iter = 1; iter <= num_iterations; iter++) {
                this->SpMV_->run();
                this->eWiseAdd_->run(this->matrix_num_rows_, 0);
                this->DenseAssign_->run(this->matrix_num_rows_, mark);
            }
            tag_iter = this->SpMV_->send_mask_device_to_host();
            this->SpMV_->send_vector_host_to_device(tag_iter);  
            this->SpMV_->send_mask_host_to_device(tot_tag);
            this->DenseAssign_->bind_mask_buf(this->SpMV_->vector_buf);
            this->DenseAssign_->bind_inout_buf(this->SpMV_->mask_buf);
            this->DenseAssign_->run(this->matrix_num_rows_, mark);
            tot_tag = this->SpMV_->send_mask_device_to_host(); 
            current_source = this->calculate_source_for_kernel(tot_tag, current_source);
        }
        return tot_tag;
    }
    aligned_dense_vec_t Fast_CC_push(uint32_t num_iterations, uint32_t num_components){
        aligned_dense_vec_t tot_tag(this->matrix_num_rows_, 0);
        int current_source = 0;
        for(size_t i = 1; i <= num_components; i++){
            aligned_dense_vec_t tag_iter(this->matrix_num_rows_, 0);
            aligned_dense_vec_t tag(this->matrix_num_rows_, 0);
            if (current_source == -1)
            {
                break;
            }
            tag[current_source] = 1;
            aligned_sparse_vec_t spmspv_input(2);
            idx_val_t head;
            graphlily::idx_t nnz = 1;  // one source vertex
            head.index = nnz;
            spmspv_input[0] = head;
            spmspv_input[1] = {current_source, 1};
            int mark = i;
            this->SpMSpV_->send_vector_host_to_device(spmspv_input);
            this->SpMSpV_->send_mask_host_to_device(tag);
            this->SparseAssign_->bind_mask_buf(this->SpMSpV_->vector_buf);
            this->SparseAssign_->bind_inout_buf(this->SpMSpV_->mask_buf);
            for (size_t iter = 1; iter <= num_iterations; iter++) {
                this->SpMSpV_->run();
                this->SpMSpV_->copy_buffer_device_to_device(
                    this->SpMSpV_->results_buf,
                    this->SpMSpV_->vector_buf,
                    sizeof(graphlily::idx_val_t) * (1 + this->SpMSpV_->get_results_nnz()));
                this->SparseAssign_->run(mark);
            }
            tag_iter = this->SpMSpV_->send_mask_device_to_host();
            this->SpMV_->send_vector_host_to_device(tag_iter);  
            this->SpMV_->send_mask_host_to_device(tot_tag);
            this->DenseAssign_->bind_mask_buf(this->SpMV_->vector_buf);
            this->DenseAssign_->bind_inout_buf(this->SpMV_->mask_buf);
            this->DenseAssign_->run(this->matrix_num_rows_, mark);
            tot_tag = this->SpMV_->send_mask_device_to_host(); 
            current_source = this->calculate_source_for_kernel(tot_tag, current_source);
        }
        return tot_tag;
    }
    aligned_dense_vec_t pull(uint32_t source, uint32_t num_iterations) {
        aligned_dense_vec_t input(this->matrix_num_rows_, semiring_.zero);
        aligned_dense_vec_t tag(this->matrix_num_rows_, 0);
        input[source] = 1;
        tag[source] = 1;
        int mark = 1;
        this->SpMV_->send_vector_host_to_device(input);
        this->SpMV_->send_mask_host_to_device(tag);
        this->DenseAssign_->bind_mask_buf(this->SpMV_->vector_buf);
        this->DenseAssign_->bind_inout_buf(this->SpMV_->mask_buf);
        this->eWiseAdd_->bind_in_buf(this->SpMV_->results_buf);
        this->eWiseAdd_->bind_out_buf(this->SpMV_->vector_buf);
        for (size_t iter = 1; iter <= num_iterations; iter++) {
            this->SpMV_->run();
            // this->SpMV_->copy_buffer_device_to_device(this->SpMV_->results_buf,
            //                                           this->SpMV_->vector_buf,
            //                                           sizeof(graphlily::val_t) * this->matrix_num_rows_);
            this->eWiseAdd_->run(this->matrix_num_rows_, 0);
            this->DenseAssign_->run(this->matrix_num_rows_, mark);
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

        // The dense tag vector
        aligned_dense_vec_t tag(this->matrix_num_rows_, 0);
        tag[source] = 1;
        int mark = 1;
        // Push
        this->SpMSpV_->send_vector_host_to_device(spmspv_input);
        this->SpMSpV_->send_mask_host_to_device(tag);
        this->SparseAssign_->bind_mask_buf(this->SpMSpV_->vector_buf);
        this->SparseAssign_->bind_inout_buf(this->SpMSpV_->mask_buf);
        for (size_t iter = 1; iter <= num_iterations; iter++) {
            this->SpMSpV_->run();
            this->SpMSpV_->copy_buffer_device_to_device(
                this->SpMSpV_->results_buf,
                this->SpMSpV_->vector_buf,
                sizeof(graphlily::idx_val_t) * (1 + this->SpMSpV_->get_results_nnz()));
            this->SparseAssign_->run(mark);
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
        spmspv_input[1] = {source, 1};

        // The dense tag vector
        aligned_dense_vec_t tag(this->matrix_num_rows_, 0);
        tag[source] = 1;

        // Push
        this->SpMSpV_->send_vector_host_to_device(spmspv_input);
        this->SpMSpV_->send_mask_host_to_device(tag);
        this->SparseAssign_->bind_mask_buf(this->SpMSpV_->vector_buf);
        this->SparseAssign_->bind_inout_buf(this->SpMSpV_->mask_buf);
        uint32_t iter = 1;
        int mark = 1;
        uint32_t vector_nnz;
        do {
            this->SpMSpV_->run();
            vector_nnz = this->SpMSpV_->get_results_nnz();
            // std::cout << "vector_nnz: " << vector_nnz << std::endl;
            this->SpMSpV_->copy_buffer_device_to_device(
                this->SpMSpV_->results_buf,
                this->SpMSpV_->vector_buf,
                sizeof(graphlily::idx_val_t) * (1 + vector_nnz));
            this->SparseAssign_->run(mark);
            iter++;
        } while (iter < num_iterations && (float(vector_nnz) / this->matrix_num_rows_ < threshold));

        std::cout << "SpMSpV runs for " << (iter - 1) << " iterations" << std::endl;

        // Switch from push to pull
        this->SpMV_->bind_mask_buf(this->SpMSpV_->mask_buf);
        aligned_sparse_vec_t spmspv_result = this->SpMSpV_->send_results_device_to_host();
        aligned_dense_vec_t spmv_input = graphlily::convert_sparse_vec_to_dense_vec<
            aligned_sparse_vec_t, aligned_dense_vec_t, graphlily::val_t>(spmspv_result,
                                                                         this->matrix_num_rows_,
                                                                         graphlily::LogicalSemiring.zero);
        this->SpMV_->send_vector_host_to_device(spmv_input);
        this->DenseAssign_->bind_mask_buf(this->SpMV_->vector_buf);
        this->DenseAssign_->bind_inout_buf(this->SpMV_->mask_buf);
        this->eWiseAdd_->bind_in_buf(this->SpMV_->results_buf);
        this->eWiseAdd_->bind_out_buf(this->SpMV_->vector_buf);

        // Pull
        for ( ; iter <= num_iterations; iter++) {
            this->SpMV_->run();
            // this->SpMV_->copy_buffer_device_to_device(
            //     this->SpMV_->results_buf,
            //     this->SpMV_->vector_buf,
            //     sizeof(graphlily::val_t) * this->matrix_num_rows_);
            this->eWiseAdd_->run(this->matrix_num_rows_, 0);
            this->DenseAssign_->run(this->matrix_num_rows_, mark);
        }

        return this->SpMSpV_->send_mask_device_to_host();  // the mask of SpMV on the host is not valid
    }


    aligned_dense_vec_t pull_push_time_breakdown(uint32_t source, uint32_t num_iterations, float threshold = 0.05) {
        float total_time_ms = 0.0;
        float spmv_spmspv_time_ms = 0.0;
        float assign_time_ms = 0.0;
        float data_transfer_time_ms = 0.0;
        // Initialize
        auto total_time_start = std::chrono::high_resolution_clock::now();
        auto spmv_spmspv_time_start = std::chrono::high_resolution_clock::now();
        auto assign_time_start = std::chrono::high_resolution_clock::now();
        auto data_transfer_time_start = std::chrono::high_resolution_clock::now();
        auto total_time_end = std::chrono::high_resolution_clock::now();
        auto spmv_spmspv_time_end = std::chrono::high_resolution_clock::now();
        auto assign_time_end = std::chrono::high_resolution_clock::now();
        auto data_transfer_time_end = std::chrono::high_resolution_clock::now();

        // The sparse input vector
        aligned_sparse_vec_t spmspv_input(2);
        idx_val_t head;
        graphlily::idx_t nnz = 1;  // one source vertex
        head.index = nnz;
        spmspv_input[0] = head;
        spmspv_input[1] = {source, 1};

        // The dense tag vector
        aligned_dense_vec_t tag(this->matrix_num_rows_, 0);
        tag[source] = 1;

        // Push
        data_transfer_time_start = std::chrono::high_resolution_clock::now();
        this->SpMSpV_->send_vector_host_to_device(spmspv_input);
        this->SpMSpV_->send_mask_host_to_device(tag);
        this->SparseAssign_->bind_mask_buf(this->SpMSpV_->vector_buf);
        this->SparseAssign_->bind_inout_buf(this->SpMSpV_->mask_buf);
        data_transfer_time_end = std::chrono::high_resolution_clock::now();
        data_transfer_time_ms += float(std::chrono::duration_cast<std::chrono::microseconds>(
            data_transfer_time_end - data_transfer_time_start).count()) / 1000;

        uint32_t iter = 1;
        int mark = 1;
        uint32_t vector_nnz;
        do {
            spmv_spmspv_time_start = std::chrono::high_resolution_clock::now();
            this->SpMSpV_->run();
            spmv_spmspv_time_end = std::chrono::high_resolution_clock::now();
            spmv_spmspv_time_ms += float(std::chrono::duration_cast<std::chrono::microseconds>(
                spmv_spmspv_time_end - spmv_spmspv_time_start).count()) / 1000;

            vector_nnz = this->SpMSpV_->get_results_nnz();

            data_transfer_time_start = std::chrono::high_resolution_clock::now();
            this->SpMSpV_->copy_buffer_device_to_device(
                this->SpMSpV_->results_buf,
                this->SpMSpV_->vector_buf,
                sizeof(graphlily::idx_val_t) * (1 + vector_nnz));
            data_transfer_time_end = std::chrono::high_resolution_clock::now();
            data_transfer_time_ms += float(std::chrono::duration_cast<std::chrono::microseconds>(
                data_transfer_time_end - data_transfer_time_start).count()) / 1000;

            assign_time_start = std::chrono::high_resolution_clock::now();
            this->SparseAssign_->run(mark);
            assign_time_end = std::chrono::high_resolution_clock::now();
            assign_time_ms += float(std::chrono::duration_cast<std::chrono::microseconds>(
                assign_time_end - assign_time_start).count()) / 1000;

            iter++;
        } while (iter < num_iterations && (float(vector_nnz) / this->matrix_num_rows_ < threshold));

        std::cout << "SpMSpV runs for " << (iter - 1) << " iterations" << std::endl;

        // Switch from push to pull
        this->SpMV_->bind_mask_buf(this->SpMSpV_->mask_buf);
        aligned_sparse_vec_t spmspv_result = this->SpMSpV_->send_results_device_to_host();
        aligned_dense_vec_t spmv_input = graphlily::convert_sparse_vec_to_dense_vec<
            aligned_sparse_vec_t, aligned_dense_vec_t, graphlily::val_t>(spmspv_result,
                                                                         this->matrix_num_rows_,
                                                                         graphlily::LogicalSemiring.zero);
        this->SpMV_->send_vector_host_to_device(spmv_input);
        this->DenseAssign_->bind_mask_buf(this->SpMV_->vector_buf);
        this->DenseAssign_->bind_inout_buf(this->SpMV_->mask_buf);
        this->eWiseAdd_->bind_in_buf(this->SpMV_->results_buf);
        this->eWiseAdd_->bind_out_buf(this->SpMV_->vector_buf);

        // Pull
        for ( ; iter <= num_iterations; iter++) {
            spmv_spmspv_time_start = std::chrono::high_resolution_clock::now();
            this->SpMV_->run();
            spmv_spmspv_time_end = std::chrono::high_resolution_clock::now();
            spmv_spmspv_time_ms += float(std::chrono::duration_cast<std::chrono::microseconds>(
                spmv_spmspv_time_end - spmv_spmspv_time_start).count()) / 1000;

            data_transfer_time_start = std::chrono::high_resolution_clock::now();
            // this->SpMV_->copy_buffer_device_to_device(
            //     this->SpMV_->results_buf,
            //     this->SpMV_->vector_buf,
            //     sizeof(graphlily::val_t) * this->matrix_num_rows_);
            this->eWiseAdd_->run(this->matrix_num_rows_, 0);
            data_transfer_time_end = std::chrono::high_resolution_clock::now();
            data_transfer_time_ms += float(std::chrono::duration_cast<std::chrono::microseconds>(
                data_transfer_time_end - data_transfer_time_start).count()) / 1000;

            assign_time_start = std::chrono::high_resolution_clock::now();
            this->DenseAssign_->run(this->matrix_num_rows_, mark);
            assign_time_end = std::chrono::high_resolution_clock::now();
            assign_time_ms += float(std::chrono::duration_cast<std::chrono::microseconds>(
                assign_time_end - assign_time_start).count()) / 1000;
        }

        data_transfer_time_start = std::chrono::high_resolution_clock::now();
        auto result = this->SpMSpV_->send_mask_device_to_host();  // the mask of SpMV on the host is not valid
        data_transfer_time_end = std::chrono::high_resolution_clock::now();
        data_transfer_time_ms += float(std::chrono::duration_cast<std::chrono::microseconds>(
            data_transfer_time_end - data_transfer_time_start).count()) / 1000;

        total_time_end = std::chrono::high_resolution_clock::now();
        total_time_ms = float(std::chrono::duration_cast<std::chrono::microseconds>(
            total_time_end - total_time_start).count()) / 1000;

        float overhead_time_ms = total_time_ms - spmv_spmspv_time_ms - assign_time_ms - data_transfer_time_ms;

        std::cout << "total_time_ms: " << total_time_ms << std::endl;
        std::cout << "spmv_spmspv_time_ms: " << spmv_spmspv_time_ms << std::endl;
        std::cout << "assign_time_ms: " << assign_time_ms << std::endl;
        std::cout << "data_transfer_time_ms: " << data_transfer_time_ms << std::endl;
        std::cout << "overhead_time_ms: " << overhead_time_ms << std::endl;

        return result;
    }


    aligned_dense_float_vec_t compute_reference_results(uint32_t num_iterations, uint32_t num_components) {
        aligned_dense_float_vec_t tot_tag(this->matrix_num_rows_, 0);
        int current_source = 0;
        for(size_t i = 1; i <= num_components; i++){
            aligned_dense_float_vec_t input(this->matrix_num_rows_, semiring_.zero);
            aligned_dense_float_vec_t tag(this->matrix_num_rows_, 0);
            input[current_source] = 1;
            tag[current_source] = 1;
            int mark = 1;
            for (size_t iter = 1; iter <= num_iterations; iter++) {
                input = this->SpMV_->compute_reference_results(input, tag);
                this->DenseAssign_->compute_reference_results(input, tag, this->matrix_num_rows_, mark);
            }
            for (size_t iter = 0; iter < tot_tag.size(); iter++){
                tot_tag[iter] = tot_tag[iter] + i * tag[iter];
            }
            current_source = this->calculate_source_for_reference(tot_tag);
            if (current_source == -1)
            {
                break;
            }
        }
        return tot_tag;
    }

    int calculate_source_for_reference(aligned_dense_float_vec_t& tot_tag){
        for(size_t i = 0; i < tot_tag.size(); i++){
            if (tot_tag[i] == 0){
                return i;
            }
        }
        return -1;
    }
    int calculate_source_for_kernel(aligned_dense_vec_t& tot_tag){
        for(size_t i = 0; i < tot_tag.size(); i++){
            if (tot_tag[i] == 0){
                return i;
            }
        }
        return -1;
    }
    int calculate_source_for_kernel(aligned_dense_vec_t& tot_tag, size_t from){
        for(size_t i = from; i < tot_tag.size(); i++){
            if (tot_tag[i] == 0){
                return i;
            }
        }
        return -1;
    }
};
}
}
#endif