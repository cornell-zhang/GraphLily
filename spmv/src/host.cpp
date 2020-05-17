#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "xcl2.hpp"

#include "kernel_spmv.h"

// HBM channels
#define MAX_HBM_CHANNEL_COUNT 32
#define CHANNEL_NAME(n) n | XCL_MEM_TOPOLOGY
const int channels[MAX_HBM_CHANNEL_COUNT] = {
    CHANNEL_NAME(0),  CHANNEL_NAME(1),  CHANNEL_NAME(2),  CHANNEL_NAME(3),  CHANNEL_NAME(4),
    CHANNEL_NAME(5),  CHANNEL_NAME(6),  CHANNEL_NAME(7),  CHANNEL_NAME(8),  CHANNEL_NAME(9),
    CHANNEL_NAME(10), CHANNEL_NAME(11), CHANNEL_NAME(12), CHANNEL_NAME(13), CHANNEL_NAME(14),
    CHANNEL_NAME(15), CHANNEL_NAME(16), CHANNEL_NAME(17), CHANNEL_NAME(18), CHANNEL_NAME(19),
    CHANNEL_NAME(20), CHANNEL_NAME(21), CHANNEL_NAME(22), CHANNEL_NAME(23), CHANNEL_NAME(24),
    CHANNEL_NAME(25), CHANNEL_NAME(26), CHANNEL_NAME(27), CHANNEL_NAME(28), CHANNEL_NAME(29),
    CHANNEL_NAME(30), CHANNEL_NAME(31)};

// Function for verifying results
bool verify(std::vector<int, aligned_allocator<int>> &reference_results,
            std::vector<int, aligned_allocator<int>> &kernel_results,
            unsigned int size) {
    bool check = true;
    for (size_t i = 0; i < size; i++) {
        if (kernel_results[i] != reference_results[i]) {
            std::cout << "Error: Result mismatch"
                      << std::endl;
            std::cout << "i = " << i
                      << " Reference result = " << reference_results[i]
                      << " Kernel result = " << kernel_results[i]
                      << std::endl;
            check = false;
            break;
        }
    }
    return check;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <XCLBIN> \n", argv[0]);
        return -1;
    }

    // Generate random sparse matrix
    unsigned int num_rows = NUM_ROWS;
    unsigned int num_cols = NUM_COLS;
    unsigned int nnz_per_row = num_cols * 0.5; // 0.05 is the sparsity

    std::vector<int, aligned_allocator<int>> indptr(num_rows + 1);
    std::vector<int, aligned_allocator<int>> indices(num_rows * nnz_per_row);
    std::vector<int, aligned_allocator<int>> vals(num_rows * nnz_per_row);

    for (size_t i = 0; i < num_rows+1; i++) {indptr[i] = i * nnz_per_row;}
    std::generate(indices.begin(), indices.end(), [&](){return std::rand() % num_cols;});
    std::generate(vals.begin(), vals.end(), [&](){return std::rand() % 256;});

    // Transform the data layout of the sparse matrix
    std::vector<int, aligned_allocator<int>> indptr_ddr(num_rows + 1);
    std::vector<int, aligned_allocator<int>> indices_hbm_0(num_rows / 2 * nnz_per_row);
    std::vector<int, aligned_allocator<int>> vals_hbm_1(num_rows / 2 * nnz_per_row);
    std::vector<int, aligned_allocator<int>> indices_hbm_2(num_rows / 2 * nnz_per_row);
    std::vector<int, aligned_allocator<int>> vals_hbm_3(num_rows / 2 * nnz_per_row);

    for (size_t i = 0; i < num_rows+1; i++) {indptr_ddr[i] = indptr[i];}

    // Be careful here
    for (size_t i = 0; i < num_rows / 2 / NUM_PE_PER_HBM_CHANNEL; i++) {
        for (size_t j = 0; j < nnz_per_row; j++) {
            for (size_t k = 0; k < NUM_PE_PER_HBM_CHANNEL; k++) {
                indices_hbm_0[i*nnz_per_row*NUM_PE_PER_HBM_CHANNEL + j*NUM_PE_PER_HBM_CHANNEL + k] =
                    indices[(i*2*NUM_PE_PER_HBM_CHANNEL + k)*nnz_per_row + j];
                vals_hbm_1[i*nnz_per_row*NUM_PE_PER_HBM_CHANNEL + j*NUM_PE_PER_HBM_CHANNEL + k] =
                    vals[(i*2*NUM_PE_PER_HBM_CHANNEL + k)*nnz_per_row + j];
                indices_hbm_2[i*nnz_per_row*NUM_PE_PER_HBM_CHANNEL + j*NUM_PE_PER_HBM_CHANNEL + k] =
                    indices[((i*2 + 1)*NUM_PE_PER_HBM_CHANNEL + k)*nnz_per_row + j];
                vals_hbm_3[i*nnz_per_row*NUM_PE_PER_HBM_CHANNEL + j*NUM_PE_PER_HBM_CHANNEL + k] =
                    vals[((i*2 + 1)*NUM_PE_PER_HBM_CHANNEL + k)*nnz_per_row + j];
            }
        }
    }

    // Initialize the dense vector randomly
    std::vector<int, aligned_allocator<int>> vector_ddr(num_cols);
    std::generate(vector_ddr.begin(), vector_ddr.end(), [&](){return std::rand() % 256;});

    // Initialize kernel_results to zero
    std::vector<int, aligned_allocator<int>> kernel_results(num_rows);
    std::fill(kernel_results.begin(), kernel_results.end(), 0);

    // Compute reference_results
    std::vector<int, aligned_allocator<int>> reference_results(num_rows);
    std::fill(reference_results.begin(), reference_results.end(), 0);
    for (size_t row_idx = 0; row_idx < num_rows; row_idx++) {
        int start = indptr[row_idx];
        int end = indptr[row_idx + 1];
        for (int i = start; i < end; i++) {
            data_t index = indices[i];
            data_t val = vals[i];
            reference_results[row_idx] += val * vector_ddr[index];
        }
    }

    // Find the OpenCL binary file
    std::string binaryFile = argv[1];
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

    std::string kernel_name = "kernel_spmv";
    /*
    std::string kernel_name_full = kernel_name + ":{" + "kernel_spmv_" + "1" + "}";
    */
    cl_int err;
    cl::CommandQueue q;
    cl::Kernel the_kernel;
    cl::Context context;

    // Find the device
    auto devices = xcl::get_xil_devices();
    int valid_device = 0;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context,
                                            device,
                                            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
                                            &err));
        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, the_kernel = cl::Kernel(program, kernel_name.c_str(), &err));
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    unsigned int num_times = 4096;
    if (xcl::is_emulation()) {
        num_times = 1;
    }

    cl_mem_ext_ptr_t vector_ddr_ext;
    cl_mem_ext_ptr_t indptr_ddr_ext;
    cl_mem_ext_ptr_t indices_hbm_0_ext;
    cl_mem_ext_ptr_t vals_hbm_1_ext;
    cl_mem_ext_ptr_t indices_hbm_2_ext;
    cl_mem_ext_ptr_t vals_hbm_3_ext;
    cl_mem_ext_ptr_t kernel_results_ext;

    cl::Buffer vector_ddr_buf;
    cl::Buffer indptr_ddr_buf;
    cl::Buffer indices_hbm_0_buf;
    cl::Buffer vals_hbm_1_buf;
    cl::Buffer indices_hbm_2_buf;
    cl::Buffer vals_hbm_3_buf;
    cl::Buffer kernel_results_buf;

    vector_ddr_ext.obj = vector_ddr.data();
    vector_ddr_ext.param = 0;

    indptr_ddr_ext.obj = indptr_ddr.data();
    indptr_ddr_ext.param = 0;

    indices_hbm_0_ext.obj = indices_hbm_0.data();
    indices_hbm_0_ext.param = 0;
    indices_hbm_0_ext.flags = channels[0];

    vals_hbm_1_ext.obj = vals_hbm_1.data();
    vals_hbm_1_ext.param = 0;
    vals_hbm_1_ext.flags = channels[1];

    indices_hbm_2_ext.obj = indices_hbm_2.data();
    indices_hbm_2_ext.param = 0;
    indices_hbm_2_ext.flags = channels[2];

    vals_hbm_3_ext.obj = vals_hbm_3.data();
    vals_hbm_3_ext.param = 0;
    vals_hbm_3_ext.flags = channels[3];

    kernel_results_ext.obj = kernel_results.data();
    kernel_results_ext.param = 0;

    // Allocate memory on the FPGA
    OCL_CHECK(err, vector_ddr_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(data_t) * num_cols,
        &vector_ddr_ext,
        &err));
    OCL_CHECK(err, indptr_ddr_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(unsigned int) * (num_rows + 1),
        &indptr_ddr_ext,
        &err));
    OCL_CHECK(err, indices_hbm_0_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(unsigned int) * (num_rows / 2 * nnz_per_row),
        &indices_hbm_0_ext,
        &err));
    OCL_CHECK(err, vals_hbm_1_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(data_t) * (num_rows / 2 * nnz_per_row),
        &vals_hbm_1_ext,
        &err));
    OCL_CHECK(err, indices_hbm_2_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(unsigned int) * (num_rows / 2 * nnz_per_row),
        &indices_hbm_2_ext,
        &err));
    OCL_CHECK(err, vals_hbm_3_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(data_t) * (num_rows / 2 * nnz_per_row),
        &vals_hbm_3_ext,
        &err));
    OCL_CHECK(err, kernel_results_buf = cl::Buffer(context,
        CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(data_t) * num_rows,
        &kernel_results_ext,
        &err));

    OCL_CHECK(err, err = the_kernel.setArg(0, vector_ddr_buf));
    OCL_CHECK(err, err = the_kernel.setArg(1, indptr_ddr_buf));
    OCL_CHECK(err, err = the_kernel.setArg(2, indices_hbm_0_buf));
    OCL_CHECK(err, err = the_kernel.setArg(3, vals_hbm_1_buf));
    OCL_CHECK(err, err = the_kernel.setArg(4, indices_hbm_2_buf));
    OCL_CHECK(err, err = the_kernel.setArg(5, vals_hbm_3_buf));
    OCL_CHECK(err, err = the_kernel.setArg(6, kernel_results_buf));
    OCL_CHECK(err, err = the_kernel.setArg(7, num_times));

    // Copy input data to Device Global Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({vector_ddr_buf,
                                                     indptr_ddr_buf,
                                                     indices_hbm_0_buf,
                                                     vals_hbm_1_buf,
                                                     indices_hbm_2_buf,
                                                     vals_hbm_3_buf}, 0 /* 0 means from host*/));
    q.finish();
    std::cout << "after enqueueMigrateMemObjects" << std::endl;

    auto kernel_start = std::chrono::high_resolution_clock::now();

    // Invoking the kernel
    OCL_CHECK(err, err = q.enqueueTask(the_kernel));
    // std::cout << "after enqueueTask" << std::endl;
    q.finish();
    // std::cout << "after finish" << std::endl;

    auto kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
    double kernel_time_in_sec = kernel_time.count();
    std::cout << "kernel_time_in_sec = " << kernel_time_in_sec << std::endl;

    // Copy result from device to host
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({kernel_results_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

    // Calculate the throughput
    double throughput = num_times * (num_rows*nnz_per_row*sizeof(data_t) // vals
                                     + num_rows*nnz_per_row*sizeof(unsigned int)); // indices
    throughput /= 1000;               // to KB
    throughput /= 1000;               // to MB
    throughput /= 1000;               // to GB
    throughput /= kernel_time_in_sec; // to GB/s
    std::cout << "THROUGHPUT = " << throughput << " GB/s" << std::endl;

    bool match = verify(reference_results, kernel_results, num_rows);
    std::cout << (match ? "TEST PASSED" : "TEST FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
