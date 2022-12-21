#include <cmath>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iostream>

#include <ap_int.h>
#include <tapa.h>

#include "graphlily.h"
#include "helper.h"

#include "io/data_loader.h"
#include "io/data_formatter.h"
#include "io/data_types.h"

#define TEST_SPMV
#include "io/data_testcases.h"

using std::cout;
using std::string;
using std::vector;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

//-------------------------------------------------------------------------
// GraphLily semiring and mask types
//-------------------------------------------------------------------------

// Operation type, named as k<opx><op+>
enum OperationType {
    kMulAdd = 0,
    kLogicalAndOr = 1,
    kAddMin = 2,
};

// Semiring definition
struct SemiringType {
    OperationType op;
    float one;  // identity element for operator <x> (a <x> one = a)
    float zero;  // identity element for operator <+> (a <+> zero = a)
};

const SemiringType ArithmeticSemiring = {kMulAdd, 1, 0};
const SemiringType LogicalSemiring = {kLogicalAndOr, 1, 0};
const SemiringType TropicalSemiring = {kAddMin, 0, FLOAT_INF};

// Mask type
enum MaskType {
    kNoMask = 0,
    kMaskWriteToZero = 1,
    kMaskWriteToOne = 2,
};


bool verify(vector<float> reference_results,
            aligned_vector<float> kernel_results,
            unsigned size) {
    float epsilon = 0.0001;

    for (size_t i = 0; i < size; i++) {
        bool match = abs(float(kernel_results[i]) - reference_results[i]) < epsilon;
        if (!match) {
            std::cout << "Error: Result mismatch"
                      << std::endl;
            std::cout << "  i = " << i
                      << "  Reference result = " << reference_results[i]
                      << "  Kernel result = " << kernel_results[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}


void compute_reference_results(
    vector<float> &mask,
    vector<float> &inout,
    uint32_t len,
    char mask_type,
    float val
) {
    if (mask_type == kMaskWriteToZero) {
        for (size_t i = 0; i < len; i++) {
            if (mask[i] == 0) {
                inout[i] = val;
            }
        }
    } else if (mask_type == kMaskWriteToOne) {
        for (size_t i = 0; i < len; i++) {
            if (mask[i] != 0) {
                inout[i] = val;
            }
        }
    } else {
        std::cout << "Invalid mask type" << std::endl;
        exit(EXIT_FAILURE);
    }
}

//---------------------------------------------------------------
// test harness
//---------------------------------------------------------------

semiring_t semi_op;
float semi_zero;
MaskType mask_type;

bool spmv_test_harness (
    std::string bitstream,
    spmv_::io::CSRMatrix<float> &mat_csr,
    bool skip_empty_rows
) {
    using namespace spmspv_::io;
    using namespace spmv_::io;

    //--------------------------------------------------------------------
    // load and format the matrix
    //--------------------------------------------------------------------
    spmv_::io::util_round_csr_matrix_dim(mat_csr, 1024, 1024);
    printf ("INFO : Test started (%d, %d)\n", semi_op, mask_type);
    CSCMatrix<float> mat_csc = csr2csc(mat_csr);

    vector<uint32_t>& CSCColPtr = mat_csc.adj_indptr;
    vector<uint32_t>& CSCRowIndex = mat_csc.adj_indices;
    vector<float>& CSCVal = mat_csc.adj_data;
    vector<uint32_t>& CSRRowPtr = mat_csr.adj_indptr;
    vector<uint32_t>& CSRColIndex = mat_csr.adj_indices;
    vector<float>& CSRVal = mat_csr.adj_data;

    uint32_t nnz = mat_csr.adj_indptr[mat_csr.num_rows];
    uint32_t M = mat_csr.num_rows, K = mat_csr.num_cols;
    float ALPHA = 1.0, BETA = 0.0;
    float EWISE_VAL = 1.0, ASSIGN_VAL = 1.0;

    // initiate vec X and vec Y
    vector<float> vec_X_cpu, vec_Y_cpu;
    vec_X_cpu.resize(K, 0.0);
    vec_Y_cpu.resize(M, 0.0);

    cout << "Generating vector X ...";
    for (int kk = 0; kk < K; ++kk) {
        vec_X_cpu[kk] = 1.0; //1.0 * (kk + 1);
    }

    cout << "Generating vector Y ...";
    for (int mm = 0; mm < M; ++mm) {
        vec_Y_cpu[mm] = -2.0 * (mm + 1);
    }

    cout << "done\n";

    cout << "Preparing sparse A for FPGA with " << NUM_CH_SPARSE << " HBM channels ...";

    vector<vector<edge> > edge_list_pes;
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

    aligned_vector<int> edge_list_ptr_fpga;
    int edge_list_ptr_fpga_size = ((edge_list_ptr.size() + 15) / 16) * 16;
    int edge_list_ptr_fpga_chunk_size = ((edge_list_ptr_fpga_size + 1023)/1024) * 1024;
    edge_list_ptr_fpga.resize(edge_list_ptr_fpga_chunk_size, semi_zero);
    for (int i = 0; i < edge_list_ptr.size(); ++i) {
        edge_list_ptr_fpga[i] = edge_list_ptr[i];
    }

    vector<aligned_vector<unsigned long> > sparse_A_fpga_vec(NUM_CH_SPARSE);
    int sparse_A_fpga_column_size = 8 * edge_list_ptr[edge_list_ptr.size()-1] * 4 / 4;
    int sparse_A_fpga_chunk_size = ((sparse_A_fpga_column_size + 511)/512) * 512;

    edge_list_64bit(edge_list_pes,
                    edge_list_ptr,
                    sparse_A_fpga_vec,
                    NUM_CH_SPARSE);

    cout << "done\n";

    cout << "Preparing vector X for FPGA ...";

    int vec_X_fpga_column_size = ((K + 16 - 1) / 16) * 16;
    int vec_X_fpga_chunk_size = ((vec_X_fpga_column_size + 1023)/1024) * 1024;
    aligned_vector<float> vec_X_fpga(vec_X_fpga_chunk_size, 0.0);

    for (int kk = 0; kk < K; ++kk) {
        vec_X_fpga[kk] = vec_X_cpu[kk];
    }

    cout << "Preparing vector Y for FPGA ...";
    int vec_Y_fpga_column_size = ((M + 16 - 1) / 16) * 16;
    int vec_Y_fpga_chunk_size = ((vec_Y_fpga_column_size + 1023)/1024) * 1024;
    aligned_vector<float> vec_Y_fpga(vec_Y_fpga_chunk_size, 0.0);
    aligned_vector<float> vec_Y_out_fpga(vec_Y_fpga_chunk_size, 0.0);

    for (int mm = 0; mm < M; ++mm) {
        vec_Y_fpga[mm] = vec_Y_cpu[mm];
    }

    cout <<  "done\n";

    cout << "Preparing vector MK for FPGA ...";

    vector<float> vec_MK_cpu(M);
    aligned_vector<float> vec_MK_fpga(vec_Y_fpga_chunk_size, 0.0);
    std::generate(vec_MK_cpu.begin(), vec_MK_cpu.end(), []{return (float)(rand() % 2);});

    for (int mm = 0; mm < M; ++mm) {
        vec_MK_fpga[mm] = vec_MK_cpu[mm];
    }

    cout <<  "done\n";

    int MAX_SIZE_edge_LIST_PTR = edge_list_ptr.size() - 1;
    int MAX_LEN_edge_PTR = edge_list_ptr[MAX_SIZE_edge_LIST_PTR];

    int * tmpPointer_v;
    tmpPointer_v = (int*) &ALPHA;
    int alpha_int = *tmpPointer_v;
    tmpPointer_v = (int*) &BETA;
    int beta_int = *tmpPointer_v;
    tmpPointer_v = (int*) &EWISE_VAL;
    int ewise_add_val_int = *tmpPointer_v;
    tmpPointer_v = (int*) &ASSIGN_VAL;
    int assign_val_int = *tmpPointer_v;

    // Actually, no need to preprocessing and just using placehold for those unused
    // arguments (e.g., vec_Y, vec_MK, etc.) is enough, but it is easier to simply
    // copy the content from `serpens_spmv.cpp`.

    cout << "launch kernel\n";
    double time_taken
    = tapa::invoke(Serpens, bitstream,
                   tapa::read_only_mmap<int>(edge_list_ptr_fpga),
                   tapa::read_only_mmaps<unsigned long, NUM_CH_SPARSE>(sparse_A_fpga_vec).reinterpret<ap_uint<512>>(),
                   tapa::read_write_mmap<float>(vec_X_fpga).reinterpret<float_v16>(),
                   tapa::read_only_mmap<float>(vec_Y_fpga).reinterpret<float_v16>(),
                   tapa::read_write_mmap<float>(vec_MK_fpga).reinterpret<float_v16>(),
                   tapa::read_write_mmap<float>(vec_Y_out_fpga).reinterpret<float_v16>(),
                   MAX_SIZE_edge_LIST_PTR,
                   MAX_LEN_edge_PTR,
                   M,
                   K,
                   alpha_int,
                   beta_int,
                   ewise_add_val_int,
                   assign_val_int,
                   semi_zero,
                   semi_op,
                   mask_type,
                   MODE_DENSE_ASSIGN
                   );
    time_taken *= 1e-9; // total time in second
    printf("Kernel time is %f ms\n", time_taken*1000);

    compute_reference_results(vec_X_cpu, vec_MK_cpu, M, mask_type, ASSIGN_VAL);

    bool pass = verify(vec_MK_cpu, vec_MK_fpga, M);
    if(pass){
        cout << "Success!\n";
    } else{
        cout << "Failed.\n";
    }
    // printf("num_mismatch = %d, percent = %.2f%%\n", mismatch_cnt, diffpercent);
    return pass;
}

//---------------------------------------------------------------
// test cases
//---------------------------------------------------------------

std::string graph_dataset_dir() { return std::string(std::getenv("DATASETS")) + "/graph/"; }
std::string nn_dataset_dir() { return std::string(std::getenv("DATASETS")) + "/pruned_nn/"; }

//---------------------------------------------------------------
// main
//---------------------------------------------------------------

int main(int argc, char **argv) {
    std::string FLAGS_bitstream = std::getenv("BITSTREAM") ? std::getenv("BITSTREAM") : "";
    assert(std::getenv("DATASETS"));
    bool passed = true;

    for (SemiringType sm : { ArithmeticSemiring, LogicalSemiring, TropicalSemiring }) {
        for (MaskType mk : { kMaskWriteToOne, kMaskWriteToZero }) {
            semi_op = sm.op;
            semi_zero = sm.zero;
            mask_type = mk;

            passed = passed && spmv_test_basic(FLAGS_bitstream);
            passed = passed && spmv_test_basic_sparse(FLAGS_bitstream);
            passed = passed && spmv_test_medium_sparse(FLAGS_bitstream);
            passed = passed && spmv_test_gplus(FLAGS_bitstream);
            // passed = passed && spmv_test_ogbl_ppa(FLAGS_bitstream);
            // passed = passed && spmv_test_hollywood(FLAGS_bitstream);
            // passed = passed && spmv_test_pokec(FLAGS_bitstream);
            // passed = passed && spmv_test_ogbn_products(FLAGS_bitstream);
            // passed = passed && spmv_test_orkut(FLAGS_bitstream);

            // passed = passed && spmv_test_transformer_50_t(FLAGS_bitstream);
            // passed = passed && spmv_test_transformer_95_t(FLAGS_bitstream);
        }
    }

    std::cout << (passed ? "===== All Test Passed! =====" : "===== Test FAILED! =====") << std::endl;
    return passed ? 0 : 1;
}
