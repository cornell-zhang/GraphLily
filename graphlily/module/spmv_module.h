#ifndef GRAPHLILY_SPMV_MODULE_H_
#define GRAPHLILY_SPMV_MODULE_H_

#include <cstdint>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>

#include "graphlily/global.h"
#include "graphlily/io/data_loader.h"
#include "graphlily/io/data_formatter.h"
#include "graphlily/module/base_module.h"

#include "graphlily/module/spmv_module_helper.h"

using graphlily::io::CSRMatrix;
using graphlily::io::CSCMatrix;

namespace graphlily {
namespace module {

template<typename matrix_data_t, typename vector_data_t>
class SpMVModule : public BaseModule {
private:
    /*! \brief The mask type */
    graphlily::MaskType mask_type_;
    /*! \brief The semiring */
    graphlily::SemiringType semiring_;
    /*! \brief The number of channels of the kernel */
    uint32_t num_channels_;
    /*! \brief The length of output buffer of the kernel */
    uint32_t out_buf_len_;
    /*! \brief The number of row partitions */
    uint32_t vec_buf_len_;

    using val_t = vector_data_t;

    using aligned_idx_t = std::vector<graphlily::idx_t, aligned_allocator<graphlily::idx_t>>;
    using aligned_dense_vec_t = std::vector<val_t, aligned_allocator<val_t>>;

    const int ARG_VEC_X = 25, ARG_VEC_Y = 26, ARG_VEC_MK = 27, ARG_VEC_Y_OUT = 28;

    const int WINDOW_SIZE = 8192;
    const int DEP_DIST_LOAD_STORE = 10;

public:
    /*! \brief Internal copy of the dense vector */
    aligned_dense_vec_t vec_X_, vec_Y_, vec_MK_;
    aligned_dense_vec_t &vector_ = vec_X_, &mask_ = vec_MK_;
    /*! \brief The kernel results */
    aligned_dense_vec_t results_;
    /*! \brief The sparse matrix using float data type in CSR format */
    CSRMatrix<float> csr_matrix_float_;
    CSCMatrix<float> csc_matrix_float_;

private:
    /*!
     * \brief The matrix data type should be the same as the vector data type.
     */
    void _check_data_type();

    aligned_vector<int> edge_list_ptr_fpga;
    vector<aligned_vector<unsigned long> > sparse_A_fpga_vec;
    uint32_t M, K;
    int MAX_SIZE_edge_LIST_PTR;
    int MAX_LEN_edge_PTR;


public:

    SpMVModule(uint32_t num_channels,
               uint32_t out_buf_len,
               uint32_t vec_buf_len) : BaseModule() {
        this->_check_data_type();
        this->num_channels_ = num_channels;
        this->out_buf_len_ = out_buf_len;
        this->vec_buf_len_ = vec_buf_len;
    }

    /* Overlay argument list:
    * Index       Argument                              Used in this module?
    * 0           Serpens edge list ptr                 y
    * 1-24        Serpens edge list channel             y
    * 25          Serpens vec X                         y
    * 26          Serpens vec Y                         y
    * 27          Serpens extended vec MK               y
    * 28          Serpens vec Y_out                     y
    *
    * 29          Serpens num iteration                 y
    * 30          Serpens num A mat length              y
    * 31          Serpens num rows (M)                  y
    * 32          Serpens num cols (K)                  y
    * 33          Serpens alpha (int cast)              y/n
    * 34          Serpens beta (int cast)               y/n
    * 35          value to add scalar                   n
    * 36          value to assign vector                n
    * 37          semiring zero                         y
    * 38          semiring operator                     y
    * 39          mask type                             y
    * 40          overlay kernel mode                   y
    */
    void set_unused_args() override {
        this->instance->SetArg(35, 0);
        this->instance->SetArg(36, 0);
    }

    void set_mode() override {
        this->instance->SetArg(40, (char)MODE_SPMV);
    }

    /*!
     * \brief Set the semiring type.
     * \param semiring The semiring type.
     */
    void set_semiring(graphlily::SemiringType semiring) {
        this->semiring_ = semiring;
    }

    /*!
     * \brief Set the mask type.
     * \param mask_type The mask type.
     */
    void set_mask_type(graphlily::MaskType mask_type) {
        this->mask_type_ = mask_type;
    }

    /*!
     * \brief Get the number of rows of the sparse matrix.
     * \return The number of rows.
     */
    uint32_t get_num_rows() {
        return this->csr_matrix_float_.num_rows;
    }

    /*!
     * \brief Get the number of columns of the sparse matrix.
     * \return The number of columns.
     */
    uint32_t get_num_cols() {
        return this->csr_matrix_float_.num_cols;
    }

    /*!
     * \brief Get the number of non-zeros of the sparse matrix.
     * \return The number of non-zeros.
     */
    uint32_t get_nnz() {
        return this->csr_matrix_float_.adj_indptr[this->csr_matrix_float_.num_rows];
    }

    /*!
     * \brief Load a csr matrix and format the csr matrix.
     *        The csr matrix should have float data type.
     *        Data type conversion, if required, is handled internally.
     * \param csr_matrix_float The csr matrix using float data type.
     * \param skip_empty_rows Whether skip empty rows.
     */
    void load_and_format_matrix(CSRMatrix<float> const &csr_matrix_float, bool skip_empty_rows);

    /*!
     * \brief Send the formatted matrix from host to device.
     */
    void send_matrix_host_to_device();

    /*!
     * \brief Send the dense vector from host to device.
     */
    void send_vector_host_to_device(aligned_dense_vec_t &vector);

    /*!
     * \brief Send the mask from host to device.
     */
    void send_mask_host_to_device(aligned_dense_vec_t &mask);

    /*!
     * \brief Bind the mask buffer to an existing buffer.
     */
    void bind_mask_buf(aligned_dense_vec_t &src_buf);

    /*!
     * \brief Run the module.
     */
    void run();

    /*!
     * \brief Send the dense vector from device to host.
     */
    aligned_dense_vec_t send_vector_device_to_host() {
        // TODO: re-enter this function
        for (int i = 0; i <= 28; ++i) {
            if (i != ARG_VEC_X) this->instance->SuspendBuf(i);
        }
        this->instance->ReadFromDevice();
        this->instance->Finish();
        return this->vec_X_;
    }

    /*!
     * \brief Send the mask from device to host.
     * \return The mask.
     */
    aligned_dense_vec_t send_mask_device_to_host() {
        for (int i = 0; i <= 28; ++i) {
            if (i != ARG_VEC_MK) this->instance->SuspendBuf(i);
        }
        this->instance->ReadFromDevice();
        this->instance->Finish();
        return this->vec_MK_;
    }

    /*!
     * \brief Send the results from device to host.
     * \return The results.
     */
    aligned_dense_vec_t send_results_device_to_host() {
        for (int i = 0; i <= 28; ++i) {
            if (i != ARG_VEC_Y_OUT) this->instance->SuspendBuf(i);
        }
        this->instance->ReadFromDevice();
        this->instance->Finish();
        return this->results_;
    }

    /*!
     * \brief Compute reference results.
     * \param vector The dense vector.
     * \return The reference results.
     */
    graphlily::aligned_dense_float_vec_t
    compute_reference_results(graphlily::aligned_dense_float_vec_t &vector);

    /*!
     * \brief Compute reference results.
     * \param vector The dense vector.
     * \param mask The mask.
     * \return The reference results.
     */
    graphlily::aligned_dense_float_vec_t
    compute_reference_results(graphlily::aligned_dense_float_vec_t &vector,
                              graphlily::aligned_dense_float_vec_t &mask);
};


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::_check_data_type() {
    assert((std::is_same<matrix_data_t, vector_data_t>::value));
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::load_and_format_matrix(
        CSRMatrix<float> const &csr_matrix_float,
        bool skip_empty_rows/*not used*/) {
    this->csr_matrix_float_ = csr_matrix_float;
    this->csc_matrix_float_ = csr2csc(csr_matrix_float);

    vector<uint32_t>& CSCColPtr = csc_matrix_float_.adj_indptr;
    vector<uint32_t>& CSCRowIndex = csc_matrix_float_.adj_indices;
    vector<float>& CSCVal = csc_matrix_float_.adj_data;
    vector<uint32_t>& CSRRowPtr = csr_matrix_float_.adj_indptr;
    vector<uint32_t>& CSRColIndex = csr_matrix_float_.adj_indices;
    vector<float>& CSRVal = csr_matrix_float_.adj_data;

    M = csr_matrix_float_.num_rows, K = csr_matrix_float_.num_cols;
    float ALPHA = 1.0, BETA = 0.0;
    assert(M % 1024 == 0 && K % 1024 == 0);

    vector<vector<edge> > edge_list_pes;
    edge_list_ptr_fpga.clear();
    vector<int> edge_list_ptr;

    generate_edge_list_for_all_PEs(CSCColPtr, //const vector<int> & CSCColPtr,
                                   CSCRowIndex, //const vector<int> & CSCRowIndex,
                                   CSCVal, //const vector<float> & CSCVal,
                                   NUM_CH_SPARSE*8, //const int NUM_PE,
                                   M, //const int NUM_ROW,
                                   K, //const int NUM_COLUMN,
                                   WINDOW_SIZE, //const int WINDOW_SIZE,
                                   edge_list_pes, //vector<vector<edge> > & edge_list_pes,
                                   edge_list_ptr, //vector<int> & edge_list_ptr,
                                   DEP_DIST_LOAD_STORE
                                   ); //const int DEP_DIST_LOAD_STORE = 10)


    int edge_list_ptr_fpga_size = ((edge_list_ptr.size() + 15) / 16) * 16;
    int edge_list_ptr_fpga_chunk_size = ((edge_list_ptr_fpga_size + 1023)/1024) * 1024;
    edge_list_ptr_fpga.resize(edge_list_ptr_fpga_chunk_size, semiring_.zero);
    for (int i = 0; i < edge_list_ptr.size(); ++i) {
        edge_list_ptr_fpga[i] = edge_list_ptr[i];
    }

    sparse_A_fpga_vec.resize(NUM_CH_SPARSE);
    int sparse_A_fpga_column_size = 8 * edge_list_ptr[edge_list_ptr.size()-1] * 4 / 4;
    int sparse_A_fpga_chunk_size = ((sparse_A_fpga_column_size + 511)/512) * 512;

    edge_list_64bit(edge_list_pes,
                    edge_list_ptr,
                    sparse_A_fpga_vec,
                    NUM_CH_SPARSE);

    MAX_SIZE_edge_LIST_PTR = edge_list_ptr.size() - 1;
    MAX_LEN_edge_PTR = edge_list_ptr[MAX_SIZE_edge_LIST_PTR];

    this->vec_X_.resize(K, semiring_.zero);
    this->vec_Y_.resize(M, 0.0);

    // Allocate result out buf
    this->results_.resize(M, 0.0);
    this->instance->SetArg(ARG_VEC_Y_OUT, frtReadWrite(this->results_));
    this->instance->WriteToDevice();
    // this->instance->Finish();

    if (this->mask_type_ == graphlily::kNoMask) {
        aligned_vector<float> tmp_f(1);
        this->instance->SetArg(ARG_VEC_MK, frtReadWrite(tmp_f));
    }
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_matrix_host_to_device() {
    this->instance->SuspendBuf(25);
    this->instance->SuspendBuf(26);
    this->instance->SuspendBuf(27);
    this->instance->SuspendBuf(28);

    this->instance->SetArg(0, frtReadWrite(this->edge_list_ptr_fpga));
    for (int i = 1; i <= 24; ++i) {
        this->instance->SetArg(i, frtReadWrite(this->sparse_A_fpga_vec[i-1]));
    }
    this->instance->WriteToDevice();
    // this->instance->Finish();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_vector_host_to_device(aligned_dense_vec_t &vector) {
    // this->vec_X_.resize(vector.size());
    this->vec_X_.assign(vector.begin(), vector.end());

    for (int i = 0; i <= 28; ++i) {
        if (i != ARG_VEC_X && i != ARG_VEC_Y) this->instance->SuspendBuf(i);
    }
    this->instance->SetArg(ARG_VEC_X, frtReadWrite(this->vec_X_));
    this->instance->SetArg(ARG_VEC_Y, frtReadWrite(this->vec_Y_));
    this->instance->WriteToDevice();
    // this->instance->Finish();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::send_mask_host_to_device(aligned_dense_vec_t &mask) {
    // this->vec_MK_.resize(mask.size());
    this->vec_MK_.assign(mask.begin(), mask.end());

    for (int i = 0; i <= 28; ++i) {
        if (i != ARG_VEC_MK) this->instance->SuspendBuf(i);
    }
    this->instance->SetArg(ARG_VEC_MK, frtReadWrite(this->vec_MK_));
    this->instance->WriteToDevice();
    // this->instance->Finish();
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::bind_mask_buf(aligned_dense_vec_t &src_buf) {
    // this->vec_MK_ = src_buf;
}


template<typename matrix_data_t, typename vector_data_t>
void SpMVModule<matrix_data_t, vector_data_t>::run() {
    this->set_mode();
    this->set_unused_args();

    // Set alpha to 1.0 and beta to 0.0, for `Y_out = Ax`
    float alpha = 1.0;
    int *alpha_ptr_int = (int*)(&alpha);
    int alpha_int = *alpha_ptr_int;

    this->instance->SetArg(29, MAX_SIZE_edge_LIST_PTR);
    this->instance->SetArg(30, MAX_LEN_edge_PTR);
    this->instance->SetArg(31, M);
    this->instance->SetArg(32, K);
    this->instance->SetArg(33, alpha_int);
    this->instance->SetArg(34, 0); // serpens beta

    this->instance->SetArg(37, semiring_.zero);
    this->instance->SetArg(38, (char)semiring_.op);
    this->instance->SetArg(39, (char)mask_type_);

    this->instance->Exec();
    this->instance->Finish();
}


#define SPMV(stmt)                                                              { \
for (size_t row_idx = 0; row_idx < this->csr_matrix_float_.num_rows; row_idx++) { \
    idx_t start = this->csr_matrix_float_.adj_indptr[row_idx];                    \
    idx_t end = this->csr_matrix_float_.adj_indptr[row_idx + 1];                  \
    for (size_t i = start; i < end; i++) {                                        \
        idx_t idx = this->csr_matrix_float_.adj_indices[i];                       \
        stmt;                                                                     \
    }                                                                             \
}                                                                               } \

template<typename matrix_data_t, typename vector_data_t>
graphlily::aligned_dense_float_vec_t
SpMVModule<matrix_data_t, vector_data_t>::compute_reference_results(aligned_dense_float_vec_t &vector) {
    float inf;
    if (std::is_same<graphlily::val_t, float>::value) {
        inf = float(graphlily::FLOAT_INF);
    } else if (std::is_same<graphlily::val_t, unsigned>::value) {
        // inf = float(graphlily::UINT_INF);
    } else {
        // inf = float(graphlily::UFIXED_INF);
    }

    aligned_dense_float_vec_t reference_results(this->csr_matrix_float_.num_rows);
    std::fill(reference_results.begin(), reference_results.end(), this->semiring_.zero);
    switch (this->semiring_.op) {
        case graphlily::kMulAdd:
            SPMV(reference_results[row_idx] += this->csr_matrix_float_.adj_data[i] * vector[idx]);
            break;
        case graphlily::kLogicalAndOr:
            SPMV(reference_results[row_idx] = reference_results[row_idx]
                || (this->csr_matrix_float_.adj_data[i] && vector[idx]));
            break;
        case graphlily::kAddMin:
            SPMV(
                // simulate the AP_SAT overflow mode
                float incr = (this->csr_matrix_float_.adj_data[i] > inf || vector[idx] > inf)
                    ? inf : std::min(this->csr_matrix_float_.adj_data[i] + vector[idx], inf);
                reference_results[row_idx] = std::min(reference_results[row_idx], incr);
            );
            break;
        default:
            std::cerr << "Invalid semiring" << std::endl;
            break;
    }
    return reference_results;
}


template<typename matrix_data_t, typename vector_data_t>
graphlily::aligned_dense_float_vec_t SpMVModule<matrix_data_t, vector_data_t>::compute_reference_results(
        graphlily::aligned_dense_float_vec_t &vector,
        graphlily::aligned_dense_float_vec_t &mask) {
    graphlily::aligned_dense_float_vec_t reference_results = this->compute_reference_results(vector);
    if (this->mask_type_ == graphlily::kMaskWriteToZero) {
        for (size_t i = 0; i < this->csr_matrix_float_.num_rows; i++) {
            if (mask[i] != 0) {
                reference_results[i] = this->semiring_.zero;
            }
        }
    } else {
        for (size_t i = 0; i < this->csr_matrix_float_.num_rows; i++) {
            if (mask[i] == 0) {
                reference_results[i] = this->semiring_.zero;
            }
        }
    }
    return reference_results;
}


}  // namespace module
}  // namespace graphlily

#endif  // GRAPHLILY_SPMV_MODULE_H_
