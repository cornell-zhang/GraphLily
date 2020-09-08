#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <ap_fixed.h>
#include "graphblas/io/data_loader.h"
#include "graphblas/io/data_formatter.h"
#include "graphblas/module/spmv_module.h"
#include "graphblas/module/spmspv_module.h"
#include "graphblas/module/assign_vector_dense_module.h"
#include "graphblas/module/add_scalar_vector_dense_module.h"


std::string target = "sw_emu";


void clean_proj_folder() {
    std::string command = "rm -rf ./" + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
}

// verify between two dense vector
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

// test two data index turple equal
template <typename val_index_type>
bool teq_dit(graphblas::index_float_struct_t a, val_index_type b) {
  float epsilon = 0.0001;
  if(abs(a.val - (float)b.val) < epsilon)
    if(a.index == b.index)
      return true;
  return false;
}

// find one dit in a dit vector
template <typename val_index_type>
bool search_dit(graphblas::aligned_index_float_struct_t vector, val_index_type element, std::vector<bool> &checklist) {
  bool found = false;
  unsigned int len = vector.size();
  for (size_t i = 0; i < len; i++) {
    if(teq_dit<val_index_type>(vector[i],element) && !checklist[i]) {
      found = true;
      checklist[i] = true;
      break;
    }
  }
  return found;
}

// verify between two sparse vector
template <typename val_index_type>
void verify_spmspv(
  graphblas::aligned_index_float_struct_t reference_result,
  std::vector<val_index_type, aligned_allocator<val_index_type>> kernel_result) {
  if (kernel_result.size() == reference_result.size()) {
    unsigned int length = reference_result.size();

    // used to keep track of whether all elements in ref is matched by result
    std::vector<bool> checklist(length,false);

    // is every element from result in reference?
    for (size_t i = 0; i < length; i++) {
      bool found_match = search_dit<val_index_type>(reference_result,kernel_result[i],checklist);
      if(!found_match) {
        std::cout << "Error: Result mismatch"
                        << std::endl;
        std::cout << "Result[" << i << "] not found in reference" << std::endl;
        std::cout << " data  = " << kernel_result[i].val   << std::endl;
        std::cout << " index = " << kernel_result[i].index << std::endl;
        exit(EXIT_FAILURE);        
      }
    }

    // is all elements in reference matched with one from result?
    for (size_t i = 0; i < length; i++) {
      if(!checklist[i]) {
        std::cout << "Error: Result mismatch"
                        << std::endl;
        std::cout << "Reference[" << i << "] not matched by results" << std::endl;
        std::cout << " data  = " << reference_result[i].val   << std::endl;
        std::cout << " index = " << reference_result[i].index << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  } else {
    std::cout << "Error: Result and Reference size Mismatch" << std::endl
              << "Result    : size " << kernel_result.size() << std::endl
              << "Reference : size " << reference_result.size() << std::endl;
    exit(EXIT_FAILURE);
  }  
}


template<typename matrix_data_t, typename vector_data_t,  graphblas::SemiRingType semiring>
void test_spmv_module() {
    uint32_t num_channels = 8;
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
    if (std::is_same<vector_data_t, ap_ufixed<32, 1>>::value) {
        for (auto &x : csr_matrix.adj_data) x = 1.0 / csr_matrix.num_rows;
    } else {
        for (auto &x : csr_matrix.adj_data) x = 1.0;
    }
    std::vector<float, aligned_allocator<float>> vector_float(csr_matrix.num_cols);
    if (std::is_same<vector_data_t, ap_ufixed<32, 1>>::value) {
         std::generate(vector_float.begin(), vector_float.end(),
            [&](){return float(rand() % 10) / 10 / csr_matrix.num_cols;});
    } else {
        std::generate(vector_float.begin(), vector_float.end(), [&](){return float(rand() % 2);});
    }
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


template<typename matrix_data_t, typename vector_data_t, graphblas::SemiRingType semiring>
void test_spmspv_module() {

  // data types
  using val_index_t = struct {
    graphblas::index_t index;
    vector_data_t val; 
  };

  using aligned_val_index_t = std::vector<val_index_t,aligned_allocator<val_index_t>>;
  using aligned_val_t = std::vector<vector_data_t,aligned_allocator<vector_data_t>>;

  // vector sparsity 99%
  float vector_sparsity = 0.99;

  // output buffer size (MUST DIVIDE 32)
  uint32_t output_buffer_len = 640000;

  // matrix data path
  std::string csc_float_npz_path = "/work/shared/common/research/graphblas/"
                                     "data/sparse_matrix_graph/uniform_100K_1000_csc_float32.npz";

  // load matrix
  struct CSCMatrix<float> csc_matrix = graphblas::io::load_csc_matrix_from_float_npz(csc_float_npz_path); 

  // generate vector
  unsigned int vector_length = csc_matrix.num_cols;
  unsigned int vector_nnz_cnt = (unsigned int)floor(vector_length * (1 - vector_sparsity));
  unsigned int vector_indices_increment = vector_length / vector_nnz_cnt;
  
  graphblas::aligned_index_float_struct_t vector_float(vector_nnz_cnt);
  for (size_t i = 0; i < vector_nnz_cnt; i++) {
    vector_float[i].val = (float)(rand() % 10) / 10;
    vector_float[i].index = i * vector_indices_increment;
  }
  graphblas::index_float_struct_t vector_head;
  vector_head.index = vector_nnz_cnt;
  vector_head.val = 0;
  vector_float.insert(vector_float.begin(),vector_head);
  aligned_val_index_t vector(vector_float.size());
  for (size_t i = 0; i < vector[0].index + 1; i++) {
    vector[i].index = vector_float[i].index;
    vector[i].val = vector_float[i].val;
  }
  

  // generate mask
  unsigned int mask_length = csc_matrix.num_rows;
  graphblas::aligned_float_t mask_float(mask_length,0);
  for (size_t i = 0; i < mask_length; i++) {
    mask_float[i] = (float)(rand() % 2);
  }
  aligned_val_t mask;
  std::copy(mask_float.begin(),mask_float.end(),std::back_inserter(mask));

  // create kernel results
  aligned_val_index_t kernel_results(csc_matrix.num_rows + 1);
  // create reference results
  graphblas::aligned_index_float_struct_t reference_results(csc_matrix.num_rows + 1);

  /*----------------------------- No mask -------------------------------*/
  {
  graphblas::module::SpMSpVModule<matrix_data_t, vector_data_t, val_index_t> module(
    semiring,output_buffer_len
  );

  module.set_target(target);
  module.set_mask_type(graphblas::kNoMask);
  module.compile();  
  module.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");
  module.load_and_format_matrix(csc_matrix);
  module.send_matrix_host_to_device();
  module.send_vector_host_to_device(vector);
  module.run();
  kernel_results = module.send_results_device_to_host();
  reference_results = module.compute_reference_results(vector_float);
  verify_spmspv<val_index_t>(reference_results,kernel_results);
  std::cout << "SpMSpV test with no mask passed" << std::endl;
  }

  /*----------------------------- With mask -------------------------------*/
  {
  graphblas::module::SpMSpVModule<matrix_data_t, vector_data_t, val_index_t> module2(
    semiring,output_buffer_len
  );
  module2.set_target(target);
  module2.set_mask_type(graphblas::kMaskWriteToZero);
  module2.compile();
  module2.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");
  module2.load_and_format_matrix(csc_matrix);
  module2.send_matrix_host_to_device();
  module2.send_mask_host_to_device(mask);
  module2.send_vector_host_to_device(vector);
  module2.run();
  kernel_results = module2.send_results_device_to_host();
  reference_results = module2.compute_reference_results(vector_float,mask_float);
  verify_spmspv<val_index_t>(reference_results,kernel_results);
  std::cout << "SpMSpV test with mask passed" << std::endl;
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


void test_add_scalar_vector_dense_module() {
    using vector_data_t = ap_ufixed<32, 1>;
    graphblas::module::eWiseAddModule<vector_data_t> module;

    uint32_t length = 128;
    vector_data_t val = 0.14;
    float val_float = float(val);

    std::vector<float, aligned_allocator<float>> in_float(length);
    std::generate(in_float.begin(), in_float.end(), [&](){return float(rand() % 10) / 100;});
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> in(in_float.begin(), in_float.end());
    std::vector<float, aligned_allocator<float>> reference_out =
        module.compute_reference_results(in_float, length, val_float);

    module.set_target(target);
    module.compile();
    module.set_up_runtime("./" + graphblas::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    module.send_in_host_to_device(in);
    module.allocate_out_buf(length);
    module.run(length, val);
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> kernel_out = module.send_out_device_to_host();
    verify<vector_data_t>(reference_out, kernel_out);

    std::cout << "eWiseAddModule test passed" << std::endl;
}


void test_copy_buffer_bind_buffer() {
    using vector_data_t = unsigned int;
    graphblas::module::AssignVectorDenseModule<vector_data_t> module;

    uint32_t length = 128;
    std::vector<float, aligned_allocator<float>> mask_float(length);
    std::generate(mask_float.begin(), mask_float.end(), [&](){return float(rand() % 2);});
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> mask(mask_float.begin(), mask_float.end());
    std::vector<float, aligned_allocator<float>> inout_float(length);
    std::fill(inout_float.begin(), inout_float.end(), 0);
    std::vector<vector_data_t, aligned_allocator<vector_data_t>> inout(inout_float.begin(), inout_float.end());

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
    // clean_proj_folder();
    // test_spmv_module<unsigned int, unsigned int, graphblas::kLogicalAndOr>();

    // clean_proj_folder();
    // test_spmv_module<ap_ufixed<32, 1>, ap_ufixed<32, 1>, graphblas::kMulAdd>();

    // clean_proj_folder();
    // test_assign_vector_dense_module();

    // clean_proj_folder();
    // test_add_scalar_vector_dense_module();

    // clean_proj_folder();
    // test_copy_buffer_bind_buffer();

    clean_proj_folder();
    test_spmspv_module<unsigned int, unsigned int, graphblas::kLogicalAndOr>();

    clean_proj_folder();
    test_spmspv_module<ap_ufixed<32, 1>, ap_ufixed<32, 1>, graphblas::kMulAdd>();
}

#pragma GCC diagnostic pop
