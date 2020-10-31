#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <ap_fixed.h>
#include "graphblas/global.h"
#include "graphblas/io/data_loader.h"
#include "graphblas/io/data_formatter.h"
#include "graphblas/module/spmv_module.h"
#include "graphblas/module/spmspv_module.h"
#include "graphblas/module/assign_vector_dense_module.h"
#include "graphblas/module/assign_vector_sparse_module.h"
#include "graphblas/module/add_scalar_vector_dense_module.h"

#include <gtest/gtest.h>

std::string target = "sw_emu";


void clean_proj_folder() {
    std::string command = "rm -rf ./" + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
}


template <typename data_t>
void verify(std::vector<float, aligned_allocator<float>> &reference_results,
            std::vector<data_t, aligned_allocator<data_t>> &kernel_results) {
    if (!(reference_results.size() == kernel_results.size())) {
        std::cout << "Size mismatch!" << std::endl;
        exit(EXIT_FAILURE);
    }
    float epsilon = 0.0001;
    for (size_t i = 0; i < reference_results.size(); i++) {
        if (abs(float(kernel_results[i]) - reference_results[i]) > epsilon) {
            std::cout << "Error: Result mismatch"
                      << std::endl;
            std::cout << "i = " << i
                      << " Reference result = " << reference_results[i]
                      << " Kernel result = " << kernel_results[i]
                      << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}



void _test_spmv_module(graphblas::module::SpMVModule<graphblas::val_t, graphblas::val_t> &module,
                       graphblas::SemiRingType semiring,
                       graphblas::MaskType mask_type,
                       CSRMatrix<float> const &csr_matrix) {
    module.set_semiring(semiring);
    module.set_mask_type(mask_type);

    std::vector<float, aligned_allocator<float>> vector_float(csr_matrix.num_cols);
    std::generate(vector_float.begin(), vector_float.end(), [&](){return float(rand() % 2);});
    std::vector<graphblas::val_t, aligned_allocator<graphblas::val_t>> vector(vector_float.begin(),
                                                                              vector_float.end());

    std::vector<float, aligned_allocator<float>> mask_float(csr_matrix.num_cols);
    std::generate(mask_float.begin(), mask_float.end(), [&](){return float(rand() % 2);});
    std::vector<graphblas::val_t, aligned_allocator<graphblas::val_t>> mask(mask_float.begin(),
                                                                            mask_float.end());

    module.load_and_format_matrix(csr_matrix);
    module.send_matrix_host_to_device();
    module.send_vector_host_to_device(vector);
    module.send_mask_host_to_device(mask);
    module.run();

    std::vector<graphblas::val_t, aligned_allocator<graphblas::val_t>> kernel_results =
        module.send_results_device_to_host();
    std::vector<float, aligned_allocator<float>> reference_results;
    if (mask_type == graphblas::kNoMask) {
        reference_results = module.compute_reference_results(vector_float);
    } else {
        reference_results = module.compute_reference_results(vector_float, mask_float);
    }

    // for (int i = 0; i < 10; i++) {
    //     std::cout << reference_results[i] << " " << kernel_results[i] << std::endl;
    // }
    // std::cout << std::endl;

    verify<graphblas::val_t>(reference_results, kernel_results);
}


TEST(SpMV, MultipleCases) {
    uint32_t out_buf_len = 512;
    uint32_t vec_buf_len = 512;
    graphblas::module::SpMVModule<graphblas::val_t, graphblas::val_t> module(graphblas::num_hbm_channels,
                                                                             out_buf_len,
                                                                             vec_buf_len);
    module.set_target(target);
    module.compile();
    module.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    std::string csr_float_npz_path = "/work/shared/common/research/graphblas/"
                                     "data/sparse_matrix_graph/dense_1K_csr_float32.npz";
    CSRMatrix<float> csr_matrix = graphblas::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
    graphblas::io::util_round_csr_matrix_dim(csr_matrix,
                                             graphblas::num_hbm_channels * graphblas::pack_size,
                                             graphblas::pack_size);
    // for (auto &x : csr_matrix.adj_data) x = 1.0 / csr_matrix.num_rows;
    for (auto &x : csr_matrix.adj_data) x = 1.0;

    _test_spmv_module(module, graphblas::kMulAdd, graphblas::kNoMask, csr_matrix);
    _test_spmv_module(module, graphblas::kLogicalAndOr, graphblas::kNoMask, csr_matrix);
    _test_spmv_module(module, graphblas::kMulAdd, graphblas::kMaskWriteToZero, csr_matrix);
    _test_spmv_module(module, graphblas::kLogicalAndOr, graphblas::kMaskWriteToZero, csr_matrix);
    _test_spmv_module(module, graphblas::kMulAdd, graphblas::kMaskWriteToOne, csr_matrix);
    _test_spmv_module(module, graphblas::kLogicalAndOr, graphblas::kMaskWriteToOne, csr_matrix);

    clean_proj_folder();
}


// template<typename matrix_data_t, typename vector_data_t, graphblas::SemiRingType semiring>
// void test_spmspv_module() {
//     // data types
//     using index_val_t = struct {graphblas::idx_t index; vector_data_t val;};
//     using aligned_sparse_vec_t = std::vector<index_val_t, aligned_allocator<index_val_t>>;
//     using aligned_dense_vec_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;

//     // vector sparsity 99%
//     float vector_sparsity = 0.99;

//     // output buffer size (MUST DIVIDE 32)
//     uint32_t out_buf_len = 128;

//     // matrix data path
//     std::string csr_float_npz_path = "/work/shared/common/research/graphblas/"
//                                      "data/sparse_matrix_graph/dense_1K_csr_float32.npz";

//     // load matrix
//     CSRMatrix<float> csr_matrix = graphblas::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
//     CSCMatrix<float> csc_matrix = graphblas::io::csr2csc(csr_matrix);

//     // generate vector
//     unsigned vector_length = csc_matrix.num_cols;
//     unsigned vector_nnz_cnt = (unsigned)floor(vector_length * (1 - vector_sparsity));
//     unsigned vector_indices_increment = vector_length / vector_nnz_cnt;

//     graphblas::aligned_sparse_float_vec_t vector_float(vector_nnz_cnt);
//     for (size_t i = 0; i < vector_nnz_cnt; i++) {
//         vector_float[i].val = (float)(rand() % 10) / 10;
//         vector_float[i].index = i * vector_indices_increment;
//     }
//     graphblas::index_float_t vector_head;
//     vector_head.index = vector_nnz_cnt;
//     vector_head.val = 0;
//     vector_float.insert(vector_float.begin(), vector_head);
//     aligned_sparse_vec_t vector(vector_float.size());
//     for (size_t i = 0; i < vector[0].index + 1; i++) {
//         vector[i].index = vector_float[i].index;
//         vector[i].val = vector_float[i].val;
//     }

//     // generate mask
//     unsigned mask_length = csc_matrix.num_rows;
//     graphblas::aligned_dense_float_vec_t mask_float(mask_length,0);
//     for (size_t i = 0; i < mask_length; i++) {
//         mask_float[i] = (float)(rand() % 2);
//     }
//     aligned_dense_vec_t mask;
//     std::copy(mask_float.begin(), mask_float.end(), std::back_inserter(mask));

//     aligned_sparse_vec_t kernel_results(csc_matrix.num_rows + 1);
//     graphblas::aligned_sparse_float_vec_t reference_results(csc_matrix.num_rows + 1);

//     /*----------------------------- No mask -------------------------------*/
//     {
//     graphblas::module::SpMSpVModule<matrix_data_t, vector_data_t, index_val_t>
//         module(semiring, out_buf_len);

//     module.set_target(target);
//     module.set_mask_type(graphblas::kNoMask);
//     module.compile();
//     module.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");
//     module.load_and_format_matrix(csc_matrix);
//     module.send_matrix_host_to_device();
//     module.send_vector_host_to_device(vector);
//     module.run();
//     kernel_results = module.send_results_device_to_host();
//     reference_results = module.compute_reference_results(vector_float);
//     aligned_dense_vec_t kernel_results_dense =
//         graphblas::convert_sparse_vec_to_dense_vec<aligned_sparse_vec_t, aligned_dense_vec_t>(
//             kernel_results, csc_matrix.num_rows);
//     graphblas::aligned_dense_float_vec_t reference_results_dense =
//         graphblas::convert_sparse_vec_to_dense_vec<graphblas::aligned_sparse_float_vec_t, graphblas::aligned_dense_float_vec_t>(
//             reference_results, csc_matrix.num_rows);
//     verify<vector_data_t>(reference_results_dense, kernel_results_dense);
//     std::cout << "SpMSpV test with no mask passed" << std::endl;
//     }

//     /*----------------------------- With mask -------------------------------*/
//     {
//     graphblas::module::SpMSpVModule<matrix_data_t, vector_data_t, index_val_t>
//         module2(semiring, out_buf_len);
//     module2.set_target(target);
//     module2.set_mask_type(graphblas::kMaskWriteToZero);
//     module2.compile();
//     module2.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");
//     module2.load_and_format_matrix(csc_matrix);
//     module2.send_matrix_host_to_device();
//     module2.send_mask_host_to_device(mask);
//     module2.send_vector_host_to_device(vector);
//     module2.run();
//     kernel_results = module2.send_results_device_to_host();
//     reference_results = module2.compute_reference_results(vector_float, mask_float);
//     aligned_dense_vec_t kernel_results_dense =
//         graphblas::convert_sparse_vec_to_dense_vec<aligned_sparse_vec_t, aligned_dense_vec_t>(
//             kernel_results, csc_matrix.num_rows);
//     graphblas::aligned_dense_float_vec_t reference_results_dense =
//         graphblas::convert_sparse_vec_to_dense_vec<graphblas::aligned_sparse_float_vec_t, graphblas::aligned_dense_float_vec_t>(
//             reference_results, csc_matrix.num_rows);
//     verify<vector_data_t>(reference_results_dense, kernel_results_dense);
//     std::cout << "SpMSpV test with mask passed" << std::endl;
//     }
// }


// void test_assign_vector_dense_module() {
//     using vector_data_t = unsigned;
//     graphblas::module::AssignVectorDenseModule<vector_data_t> module;

//     uint32_t length = 128;
//     vector_data_t val = 23;
//     float val_float = float(val);

//     std::vector<float, aligned_allocator<float>> mask_float(length);
//     std::generate(mask_float.begin(), mask_float.end(), [&](){return float(rand() % 2);});
//     std::vector<vector_data_t, aligned_allocator<vector_data_t>> mask(mask_float.begin(), mask_float.end());

//     std::vector<float, aligned_allocator<float>> reference_inout(length);
//     std::generate(reference_inout.begin(), reference_inout.end(), [&](){return float(rand() % 128);});
//     std::vector<vector_data_t, aligned_allocator<vector_data_t>> kernel_inout(reference_inout.begin(),
//                                                                               reference_inout.end());

//     module.set_target(target);
//     module.set_mask_type(graphblas::kMaskWriteToOne);
//     module.compile();
//     module.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

//     module.send_mask_host_to_device(mask);
//     module.send_inout_host_to_device(kernel_inout);
//     module.run(length, val);
//     kernel_inout = module.send_inout_device_to_host();
//     module.compute_reference_results(mask_float, reference_inout, length, val_float);
//     verify<vector_data_t>(reference_inout, kernel_inout);

//     std::cout << "AssignVectorDenseModule test passed" << std::endl;
// }


// void test_assign_vector_sparse_module() {
//     using vector_data_t = unsigned;
//     using sparse_data_t = struct {
//         graphblas::idx_t index;
//         vector_data_t val;
//     };

//     // mask sparsity 90%
//     float mask_sparsity = 0.9;

//     uint32_t inout_size = 8192;
//     vector_data_t val = 7216;
//     float val_float = float(val);
//     float f_uint_inf = float(graphblas::UINT_INF);

//     unsigned length = (unsigned)floor(inout_size * (1 - mask_sparsity));
//     unsigned mask_indices_increment = inout_size / length;

//     graphblas::module::AssignVectorSparseModule<vector_data_t,sparse_data_t> module;
//     module.set_target(target);
//     module.compile();
//     module.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

//     /*----------------------------- Mode 0 (BFS) -------------------------------*/

//     graphblas::aligned_sparse_float_vec_t mask_float_bfs(length + 1);
//     for (size_t i = 0; i < length; i++) {
//         mask_float_bfs[i+1].val = float(rand() % 10);
//         mask_float_bfs[i+1].index = i * mask_indices_increment;
//     }
//     mask_float_bfs[0].val = 0;
//     mask_float_bfs[0].index = length;
//     std::vector<sparse_data_t,aligned_allocator<sparse_data_t>> mask_bfs(length + 1);
//     for (size_t i = 0; i < length + 1; i++) {
//         mask_bfs[i].val = mask_float_bfs[i].val;
//         mask_bfs[i].index = mask_float_bfs[i].index;
//     }

//     graphblas::aligned_dense_float_vec_t reference_inout_bfs(inout_size);
//     std::generate(reference_inout_bfs.begin(), reference_inout_bfs.end(), [&](){return (rand() % 10);});
//     std::vector<vector_data_t,aligned_allocator<vector_data_t>> kernel_inout_bfs(reference_inout_bfs.begin(), reference_inout_bfs.end());
//     graphblas::aligned_sparse_float_vec_t reference_dummy_nf;

//     module.send_mask_host_to_device(mask_bfs);
//     module.send_inout_host_to_device(kernel_inout_bfs);
//     module.set_mode(0);
//     module.run(val);
//     kernel_inout_bfs = module.send_inout_device_to_host();
//     module.compute_reference_results(mask_float_bfs, reference_inout_bfs, reference_dummy_nf, val_float);
//     verify<vector_data_t>(reference_inout_bfs, kernel_inout_bfs);

//     std::cout << "AssignVectorSparseModule BFS test passed" << std::endl;

//     /*----------------------------- Mode 1 (SSSP) -------------------------------*/

//     graphblas::aligned_sparse_float_vec_t mask_float_sssp(length + 1);
//     for (size_t i = 0; i < length; i++) {
//         mask_float_sssp[i+1].val = float(rand() % 10);
//         mask_float_sssp[i+1].index = i * mask_indices_increment;
//     }
//     mask_float_sssp[0].val = 0;
//     mask_float_sssp[0].index = length;
//     std::vector<sparse_data_t,aligned_allocator<sparse_data_t>> mask_sssp(length + 1);
//     for (size_t i = 0; i < length + 1; i++) {
//         mask_sssp[i].val = mask_float_sssp[i].val;
//         mask_sssp[i].index = mask_float_sssp[i].index;
//     }

//     graphblas::aligned_dense_float_vec_t reference_inout_sssp(inout_size);
//     std::generate(reference_inout_sssp.begin(), reference_inout_sssp.end(), [&](){return (((rand() % 10) > 5) ? 3 : f_uint_inf);});
//     std::vector<vector_data_t,aligned_allocator<vector_data_t>> kernel_inout_sssp(inout_size);
//     for (size_t i = 0; i < reference_inout_sssp.size() + 1; i++) {
//         if(reference_inout_sssp[i] == f_uint_inf) {
//             kernel_inout_sssp[i] = graphblas::UINT_INF;
//         } else {
//             kernel_inout_sssp[i] = reference_inout_sssp[i];
//         }
//     }

//     graphblas::aligned_sparse_float_vec_t reference_new_frontier;
//     std::vector<sparse_data_t,aligned_allocator<sparse_data_t>> kernel_new_frontier;
//     std::vector<vector_data_t,aligned_allocator<vector_data_t>> kernel_inout_print(kernel_inout_sssp.begin(),kernel_inout_sssp.end());

//     module.send_mask_host_to_device(mask_sssp);
//     module.send_inout_host_to_device(kernel_inout_sssp);
//     module.set_mode(1);
//     module.run(0);
//     kernel_inout_sssp = module.send_inout_device_to_host();
//     kernel_new_frontier = module.send_new_frontier_device_to_host();
//     module.compute_reference_results(mask_float_sssp, reference_inout_sssp, reference_new_frontier, 0);

//     verify<vector_data_t>(reference_inout_sssp, kernel_inout_sssp);
//     std::cout << "[INFO test_assign_sparse:SSSP] Inout Matched" << std::endl;

//     graphblas::aligned_dense_float_vec_t dense_ref_nf =
//         graphblas::convert_sparse_vec_to_dense_vec<
//             graphblas::aligned_sparse_float_vec_t,graphblas::aligned_dense_float_vec_t
//         >(reference_new_frontier,inout_size);

//     std::vector<vector_data_t,aligned_allocator<vector_data_t>> dense_knl_nf =
//         graphblas::convert_sparse_vec_to_dense_vec<
//             std::vector<sparse_data_t,aligned_allocator<sparse_data_t>>,
//             std::vector<vector_data_t,aligned_allocator<vector_data_t>>
//         >(kernel_new_frontier,inout_size);

//     verify<vector_data_t>(dense_ref_nf, dense_knl_nf);
//     std::cout << "[INFO test_assign_sparse:SSSP] New_frontier Matched" << std::endl;
//     std::cout << "AssignVectorSparseModule SSSP test passed" << std::endl;
// }


// void test_add_scalar_vector_dense_module() {
//     using vector_data_t = ap_ufixed<32, 1>;
//     graphblas::module::eWiseAddModule<vector_data_t> module;

//     uint32_t length = 128;
//     vector_data_t val = 0.14;
//     float val_float = float(val);

//     std::vector<float, aligned_allocator<float>> in_float(length);
//     std::generate(in_float.begin(), in_float.end(), [&](){return float(rand() % 10) / 100;});
//     std::vector<vector_data_t, aligned_allocator<vector_data_t>> in(in_float.begin(), in_float.end());
//     std::vector<float, aligned_allocator<float>> reference_out =
//         module.compute_reference_results(in_float, length, val_float);

//     module.set_target(target);
//     module.compile();
//     module.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

//     module.send_in_host_to_device(in);
//     module.allocate_out_buf(length);
//     module.run(length, val);
//     std::vector<vector_data_t, aligned_allocator<vector_data_t>> kernel_out = module.send_out_device_to_host();
//     verify<vector_data_t>(reference_out, kernel_out);

//     std::cout << "eWiseAddModule test passed" << std::endl;
// }


// void test_copy_buffer_bind_buffer() {
//     using vector_data_t = unsigned;
//     graphblas::module::AssignVectorDenseModule<vector_data_t> module;

//     uint32_t length = 128;
//     std::vector<float, aligned_allocator<float>> mask_float(length);
//     std::generate(mask_float.begin(), mask_float.end(), [&](){return float(rand() % 2);});
//     std::vector<vector_data_t, aligned_allocator<vector_data_t>> mask(mask_float.begin(), mask_float.end());
//     std::vector<float, aligned_allocator<float>> inout_float(length);
//     std::fill(inout_float.begin(), inout_float.end(), 0);
//     std::vector<vector_data_t, aligned_allocator<vector_data_t>> inout(inout_float.begin(), inout_float.end());

//     module.set_target(target);
//     module.set_mask_type(graphblas::kMaskWriteToOne);
//     module.compile();
//     module.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

//     /*----------------------------- Copy buffer -------------------------------*/
//     {
//     module.send_mask_host_to_device(mask);
//     module.send_inout_host_to_device(inout);
//     module.copy_buffer_device_to_device(module.mask_buf, module.inout_buf, sizeof(vector_data_t) * length);
//     inout = module.send_inout_device_to_host();
//     verify<vector_data_t>(mask_float, inout);
//     std::cout << "CopyBuffer test passed" << std::endl;
//     }

//     /*----------------------------- Bind buffer -------------------------------*/
//     {
//     std::vector<float, aligned_allocator<float>> x_float(length);
//     std::fill(x_float.begin(), x_float.end(), 0);
//     std::vector<vector_data_t, aligned_allocator<vector_data_t>> x(x_float.begin(), x_float.end());
//     cl_mem_ext_ptr_t x_ext;
//     x_ext.obj = x.data();
//     x_ext.param = 0;
//     x_ext.flags = graphblas::DDR[0];
//     cl::Device device = graphblas::find_device();
//     cl::Context context = cl::Context(device, NULL, NULL, NULL);
//     cl::Buffer x_buf = cl::Buffer(context,
//                                   CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
//                                   sizeof(vector_data_t) * length,
//                                   &x_ext);
//     cl::CommandQueue command_queue = cl::CommandQueue(context, device);

//     module.send_mask_host_to_device(mask);
//     module.bind_inout_buf(x_buf);
//     module.run(length, 2);
//     command_queue.enqueueMigrateMemObjects({x_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
//     command_queue.finish();

//     module.compute_reference_results(mask_float, inout_float, length, 2);
//     verify<vector_data_t>(inout_float, x);
//     std::cout << "BindBuffer test passed" << std::endl;
//     }
// }


int main(int argc, char ** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#pragma GCC diagnostic pop
