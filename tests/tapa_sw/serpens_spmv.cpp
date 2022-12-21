#include <cmath>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <thread>

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
using std::min;

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
    // assert(M == K);

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

    std::thread device_compute([&]{
        cout << "Run spmv on cpu...";
        auto start_cpu = std::chrono::steady_clock::now();
        cpu_spmv_CSR(M, K, nnz, ALPHA,
                    CSRRowPtr,
                    CSRColIndex,
                    CSRVal,
                    vec_X_cpu,
                    BETA,
                    vec_Y_cpu,
                    semi_op,
                    semi_zero,
                    mask_type,
                    vec_MK_cpu);
        auto end_cpu = std::chrono::steady_clock::now();
        double time_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count();
        time_cpu *= 1e-9;
        cout << "done (" << time_cpu*1000 << " msec)\n";
        cout <<"CPU GFLOPS: " << 2.0 * (nnz + M) / 1e+9 / time_cpu << "\n";
    });

    int MAX_SIZE_edge_LIST_PTR = edge_list_ptr.size() - 1;
    int MAX_LEN_edge_PTR = edge_list_ptr[MAX_SIZE_edge_LIST_PTR];

    int * tmpPointer_v;
    tmpPointer_v = (int*) &ALPHA;
    int alpha_int = *tmpPointer_v;
    tmpPointer_v = (int*) &BETA;
    int beta_int = *tmpPointer_v;

    cout << "launch kernel\n";

    double time_taken
    = tapa::invoke(Serpens, bitstream,
                   tapa::read_only_mmap<int>(edge_list_ptr_fpga),
                   tapa::read_only_mmaps<unsigned long, NUM_CH_SPARSE>(sparse_A_fpga_vec).reinterpret<ap_uint<512>>(),
                   tapa::read_only_mmap<float>(vec_X_fpga).reinterpret<float_v16>(),
                   tapa::read_only_mmap<float>(vec_Y_fpga).reinterpret<float_v16>(),
                   tapa::read_only_mmap<float>(vec_MK_fpga).reinterpret<float_v16>(),
                   tapa::write_only_mmap<float>(vec_Y_out_fpga).reinterpret<float_v16>(),
                   MAX_SIZE_edge_LIST_PTR,
                   MAX_LEN_edge_PTR,
                   M,
                   K,
                   alpha_int,
                   beta_int,
                   0,
                   0,
                   semi_zero,
                   semi_op,
                   mask_type,
                   MODE_SPMV
                   );

    // Use fpga-runtime APIs:
    /*
    #define buffer(x) fpga::Placeholder((x).data(), (x).size())
    #define enable(x) fpga::ReadWrite((x).data(), (x).size())
    fpga::Instance inst(bitstream);

    auto h2d_1 = std::chrono::high_resolution_clock::now();

    inst.SetArg(0, buffer(edge_list_ptr_fpga));
    for (int i = 1; i <= 24; ++i) inst.SetArg(i, buffer(sparse_A_fpga_vec[i-1]));
    inst.SetArg(25, enable(vec_X_fpga));
    inst.SetArg(26, buffer(vec_Y_fpga));
    inst.SetArg(27, buffer(vec_MK_fpga));
    inst.SetArg(28, enable(vec_Y_out_fpga));

    inst.WriteToDevice();
    inst.Finish();

    auto h2d_2 = std::chrono::high_resolution_clock::now();

    const int num_runs = 1024;
    double frt_time_taken_ms = 0.0, std_chrono_time_taken_ms = 0.0;

    for (int i = 0; i < num_runs; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        inst.SetArg(29, MAX_SIZE_edge_LIST_PTR);
        inst.SetArg(30, MAX_LEN_edge_PTR);
        inst.SetArg(31, M);
        inst.SetArg(32, K);
        inst.SetArg(33, alpha_int);
        inst.SetArg(34, beta_int);
        inst.SetArg(35, i);
        inst.SetArg(36, i);
        inst.SetArg(37, semi_zero);
        inst.SetArg(38, (char)semi_op);
        inst.SetArg(39, (char)mask_type);
        inst.SetArg(40, (char)MODE_SPMV);

        inst.Exec();
        inst.Finish();
        auto t2 = std::chrono::high_resolution_clock::now();
        std_chrono_time_taken_ms += double(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
        frt_time_taken_ms += inst.ComputeTimeSeconds();
    }

    std_chrono_time_taken_ms = std_chrono_time_taken_ms / 1000 / num_runs;
    frt_time_taken_ms = frt_time_taken_ms * 1000 / num_runs;

    auto d2h_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i <= 27; ++i) inst.SuspendBuf(i);
    inst.ReadFromDevice();
    inst.Finish();
    auto d2h_2 = std::chrono::high_resolution_clock::now();

    double transfer_time_ms = (double(std::chrono::duration_cast<std::chrono::microseconds>(h2d_2 - h2d_1).count()) +
                               double(std::chrono::duration_cast<std::chrono::microseconds>(d2h_2 - d2h_1).count())) / 1000;

    printf(">>> Kernel time is (frt %f, std %f) ms\n", frt_time_taken_ms, std_chrono_time_taken_ms);
    printf(">>> Data Transfer time is %f ms\n", transfer_time_ms);
    */

    time_taken *= 1e-9; // total time in second
    printf(">>> Kernel time: %f \n", time_taken*1000);

    float gflops =
    2.0 * (nnz + M)
    / 1e+9
    / time_taken
    ;
    printf(">>> GFLOPS:%f \n", gflops);

    device_compute.join();
    bool pass = verify(vec_Y_cpu, vec_Y_out_fpga, M);
    if(pass){
        cout << "Success!\n";
    } else{
        cout << "Failed.\n";
    }
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
        for (MaskType mk : { kNoMask, kMaskWriteToOne, kMaskWriteToZero }) {
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
