#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include "ap_fixed.h"
#include "graphblas/io/data_loader.h"
#include "graphblas/io/data_formatter.h"
#include "graphblas/module/spmv_module.h"
#include "graphblas/module/assign_vector_dense_module.h"


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


void test_spmv_module() {
    graphblas::SemiRingType semiring = graphblas::kLogicalAndOr;
    uint32_t num_channels = 16;
    using matrix_data_t = bool;
    using vector_data_t = unsigned int; // Use unsigned int to work around the issue with std::vector<bool>
    std::string target = "sw_emu";

    uint32_t out_buffer_len;
    uint32_t vector_buffer_len;
    if (target == "hw") {
        out_buffer_len = 5120;
        vector_buffer_len = 5120;
    } else {
        // Avoid stack overflow by using small arrays in emulation.
        out_buffer_len = 512;
        vector_buffer_len = 512;
    }

    std::string csr_float_npz_path = "/work/shared/common/research/graphblas/"
                                     "data/sparse_matrix_graph/dense_1K_csr_float32.npz";
    struct CSRMatrix<float> csr_matrix = graphblas::io::load_csr_matrix_from_float_npz(csr_float_npz_path);
    graphblas::io::util_round_csr_matrix_dim(csr_matrix,
                                             num_channels * graphblas::pack_size,
                                             graphblas::pack_size);
    std::vector<float, aligned_allocator<float>> vector_float(csr_matrix.num_cols);
    std::generate(vector_float.begin(), vector_float.end(), [&](){return float(rand() % 2);});
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> vector(vector_float.begin(),
                                                                        vector_float.end());
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> kernel_results;
    std::vector<float, aligned_allocator<float>> reference_results;

    /*----------------------------- No mask -------------------------------*/
    {
    graphblas::module::SpMVModule<matrix_data_t, vector_data_t> module1(semiring,
                                                                        num_channels,
                                                                        out_buffer_len,
                                                                        vector_buffer_len);
    module1.set_target(target);
    module1.set_mask_type(graphblas::kNoMask);
    module1.compile();
    module1.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    module1.load_and_format_matrix(csr_matrix);
    module1.send_matrix_host_to_device();
    module1.send_vector_host_to_device(vector);
    module1.run();
    kernel_results = module1.send_results_device_to_host();
    reference_results = module1.compute_reference_results(vector_float);
    verify<vector_data_t>(reference_results, kernel_results);

    std::cout << "SpMV test with no mask passed" << std::endl;
    }

    clean_proj_folder();

    /*----------------------------- Use mask -------------------------------*/
    {
    graphblas::module::SpMVModule<matrix_data_t, vector_data_t> module2(semiring,
                                                                        num_channels,
                                                                        out_buffer_len,
                                                                        vector_buffer_len);
    module2.set_target(target);
    module2.set_mask_type(graphblas::kMaskWriteToZero);
    module2.compile();
    module2.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    module2.load_and_format_matrix(csr_matrix);
    module2.send_matrix_host_to_device();

    std::vector<float, aligned_allocator<float>> mask_float(csr_matrix.num_cols);
    std::generate(mask_float.begin(), mask_float.end(), [&](){return float(rand() % 2);});
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> mask(mask_float.begin(), mask_float.end());

    module2.send_vector_host_to_device(vector);
    module2.send_mask_host_to_device(mask);
    module2.run();
    kernel_results = module2.send_results_device_to_host();
    reference_results = module2.compute_reference_results(vector_float, mask_float);
    verify<vector_data_t>(reference_results, kernel_results);

    std::cout << "SpMV test with mask passed" << std::endl;
    }
}


void test_assign_vector_dense_module() {
    using vector_data_t = unsigned int;
    graphblas::module::AssignVectorDenseModule<vector_data_t> module;

    uint32_t length = 128;
    vector_data_t val = 23;
    float val_float = float(val);

    std::vector<float, aligned_allocator<float>> mask_float(length);
    std::generate(mask_float.begin(), mask_float.end(), [&](){return float(rand() % 2);});
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> mask(mask_float.begin(), mask_float.end());

    std::vector<float, aligned_allocator<float>> reference_inout(length);
    std::generate(reference_inout.begin(), reference_inout.end(), [&](){return float(rand() % 128);});
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> kernel_inout(reference_inout.begin(),
                                                                              reference_inout.end());

    std::string target = "sw_emu";
    module.set_target(target);
    module.set_mask_type(graphblas::kMaskWriteToOne);
    module.compile();
    module.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    module.send_mask_host_to_device(mask);
    module.send_inout_host_to_device(kernel_inout);
    module.run(length, val);
    kernel_inout = module.send_inout_device_to_host();
    module.compute_reference_results(mask_float, reference_inout, length, val_float);
    verify<vector_data_t>(reference_inout, kernel_inout);

    std::cout << "AssignVectorDenseModule test passed" << std::endl;
}


void test_copy_buffer_bind_buffer() {
    using vector_data_t = unsigned int;
    graphblas::module::AssignVectorDenseModule<vector_data_t> module;

    uint32_t length = 10;
    std::vector<float, aligned_allocator<float>> mask_float(length);
    std::generate(mask_float.begin(), mask_float.end(), [&](){return float(rand() % 2);});
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> mask(mask_float.begin(), mask_float.end());
    std::vector<float, aligned_allocator<float>> inout_float(length);
    std::fill(inout_float.begin(), inout_float.end(), 0);
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> inout(inout_float.begin(), inout_float.end());

    std::string target = "sw_emu";
    module.set_target(target);
    module.set_mask_type(graphblas::kMaskWriteToOne);
    module.compile();
    module.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    /*----------------------------- Copy buffer -------------------------------*/
    {
    module.send_mask_host_to_device(mask);
    module.send_inout_host_to_device(inout);
    module.copy_buffer_device_to_device(module.mask_buf, module.inout_buf, sizeof(vector_data_t) * length);
    inout = module.send_inout_device_to_host();
    verify<vector_data_t>(mask_float, inout);
    std::cout << "CopyBuffer test passed" << std::endl;
    }

    /*----------------------------- Bind buffer -------------------------------*/
    {
    std::vector<float, aligned_allocator<float>> x_float(length);
    std::fill(x_float.begin(), x_float.end(), 0);
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> x(x_float.begin(), x_float.end());
    cl_mem_ext_ptr_t x_ext;
    x_ext.obj = x.data();
    x_ext.param = 0;
    x_ext.flags = graphblas::DDR[0];
    cl::Device device = graphblas::find_device();
    cl::Context context = cl::Context(device, NULL, NULL, NULL);
    cl::Buffer x_buf = cl::Buffer(context,
                                  CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                  sizeof(vector_data_t) * length,
                                  &x_ext);
    cl::CommandQueue command_queue = cl::CommandQueue(context, device);

    module.send_mask_host_to_device(mask);
    module.bind_inout_buf(x_buf);
    module.run(length, 2);
    command_queue.enqueueMigrateMemObjects({x_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    command_queue.finish();

    module.compute_reference_results(mask_float, inout_float, length, 2);
    verify<vector_data_t>(inout_float, x);
    std::cout << "BindBuffer test passed" << std::endl;
    }
}


int main(int argc, char *argv[]) {
    clean_proj_folder();
    test_spmv_module();

    clean_proj_folder();
    test_assign_vector_dense_module();

    clean_proj_folder();
    test_copy_buffer_bind_buffer();
}

#pragma GCC diagnostic pop
