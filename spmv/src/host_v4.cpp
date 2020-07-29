#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"

#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>

#include "xcl2.hpp"

#include "kernel_spmv_v4.h"
#include "graph_partitioning.h"

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

const int ddr[2] = {CHANNEL_NAME(32),  CHANNEL_NAME(33)};

// Function for verifying results
bool verify(std::vector<float, aligned_allocator<float>> &reference_results,
            std::vector<data_t, aligned_allocator<data_t>> &kernel_results,
            unsigned int size) {
    float epsilon = 0.0001;
    bool check = true;
    for (size_t i = 0; i < size; i++) {
        if (abs(float(kernel_results[i]) - reference_results[i]) > epsilon) {
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

    using packed_data_t = struct {data_t data[VDATA_SIZE];};

    // Data loading and formatting
    SpMVDataFormatter<data_t, NUM_PE_PER_HBM_CHANNEL, packed_data_t, packed_index_t>
        formatter("/work/shared/common/research/graphblas/data/sparse_matrix_graph/uniform_10K_10_csr_float32.npz");

    std::cout << "Finished loading data" << std::endl;

    unsigned int num_rows = formatter.get_num_rows();
    unsigned int num_cols = formatter.get_num_cols();
    std::vector<unsigned int> indptr = formatter.get_indptr();
    std::vector<unsigned int> indices = formatter.get_indices();
    unsigned int nnz = indptr[num_rows];

    std::cout << "num_rows: " << num_rows << std::endl;
    std::cout << "num_cols: " << num_cols << std::endl;
    std::cout << "nnz: " << nnz << std::endl;

    formatter.format_pad_marker_end_of_row(VECTOR_BUFFER_LEN, NUM_HBM_CHANNEL, VAL_MARKER, IDX_MARKER);

    std::cout << "Finished formatting data" << std::endl;

    unsigned int num_col_partitions = (num_cols + VECTOR_BUFFER_LEN - 1) / VECTOR_BUFFER_LEN;

    std::vector<packed_index_t, aligned_allocator<packed_index_t>> channel_0_indices;
    std::vector<unsigned int, aligned_allocator<unsigned int>> channel_0_partition_indptr(num_col_partitions + 1);
    channel_0_partition_indptr[0] = 0;
    for (size_t i = 0; i < num_col_partitions; i++) {
        auto channel_0_indices_partition = formatter.get_packed_indices(i, 0);
        channel_0_partition_indptr[i + 1] = channel_0_partition_indptr[i] + channel_0_indices_partition.size();
        channel_0_indices.insert(channel_0_indices.end(), channel_0_indices_partition.begin(), channel_0_indices_partition.end());
    }

    std::vector<packed_index_t, aligned_allocator<packed_index_t>> channel_1_indices;
    std::vector<unsigned int, aligned_allocator<unsigned int>> channel_1_partition_indptr(num_col_partitions + 1);
    channel_1_partition_indptr[0] = 0;
    for (size_t i = 0; i < num_col_partitions; i++) {
        auto channel_1_indices_partition = formatter.get_packed_indices(i, 1);
        channel_1_partition_indptr[i + 1] = channel_1_partition_indptr[i] + channel_1_indices_partition.size();
        channel_1_indices.insert(channel_1_indices.end(), channel_1_indices_partition.begin(), channel_1_indices_partition.end());
    }

    // Initialize the dense vector randomly
    std::vector<float, aligned_allocator<float>> vector_float(num_cols);
    std::generate(vector_float.begin(), vector_float.end(), [&](){return float(rand() % num_cols) / num_cols / num_cols;});
    std::vector<data_t, aligned_allocator<data_t>> vector(num_cols);
    for (size_t i = 0; i < num_cols; i++) vector[i] = data_t(vector_float[i]); // data type casting

    // Initialize kernel_results to zero
    std::vector<data_t, aligned_allocator<data_t>> kernel_results(num_rows);
    std::fill(kernel_results.begin(), kernel_results.end(), 0);

    // Compute reference_results
    std::vector<float, aligned_allocator<float>> reference_results(num_rows);
    std::fill(reference_results.begin(), reference_results.end(), 0);
    for (size_t row_idx = 0; row_idx < num_rows; row_idx++) {
        int start = indptr[row_idx];
        int end = indptr[row_idx + 1];
        for (int i = start; i < end; i++) {
            unsigned int index = indices[i];
            reference_results[row_idx] += vector_float[index];
        }
    }

    std::cout << "Finished computing reference results" << std::endl;

    // Find the OpenCL binary file
    std::string binaryFile = argv[1];
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

    std::string kernel_name = "kernel_spmv_v4";
    /*
    std::string kernel_name_full = kernel_name + ":{" + "kernel_spmv_v4_" + "1" + "}";
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

    unsigned int num_times = 1;
    if (xcl::is_emulation()) {
        num_times = 2;
    }

    cl_mem_ext_ptr_t vector_ext;
    cl_mem_ext_ptr_t channel_0_partition_indptr_ext;
    cl_mem_ext_ptr_t channel_0_indices_ext;
    cl_mem_ext_ptr_t channel_1_partition_indptr_ext;
    cl_mem_ext_ptr_t channel_1_indices_ext;
    cl_mem_ext_ptr_t kernel_results_ext;

    cl::Buffer vector_buf;
    cl::Buffer channel_0_partition_indptr_buf;
    cl::Buffer channel_0_indices_buf;
    cl::Buffer channel_1_partition_indptr_buf;
    cl::Buffer channel_1_indices_buf;
    cl::Buffer kernel_results_buf;

    vector_ext.obj = vector.data();
    vector_ext.param = 0;
    vector_ext.flags = ddr[0];

    channel_0_partition_indptr_ext.obj = channel_0_partition_indptr.data();
    channel_0_partition_indptr_ext.param = 0;
    channel_0_partition_indptr_ext.flags = ddr[0];

    channel_0_indices_ext.obj = channel_0_indices.data();
    channel_0_indices_ext.param = 0;
    channel_0_indices_ext.flags = channels[0];

    channel_1_partition_indptr_ext.obj = channel_1_partition_indptr.data();
    channel_1_partition_indptr_ext.param = 0;
    channel_1_partition_indptr_ext.flags = ddr[0];

    channel_1_indices_ext.obj = channel_1_indices.data();
    channel_1_indices_ext.param = 0;
    channel_1_indices_ext.flags = channels[1];

    kernel_results_ext.obj = kernel_results.data();
    kernel_results_ext.param = 0;
    kernel_results_ext.flags = ddr[1];

    // Allocate memory on the FPGA
    OCL_CHECK(err, vector_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(data_t) * num_cols,
        &vector_ext,
        &err));
    OCL_CHECK(err, channel_0_partition_indptr_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(unsigned int) * (num_col_partitions + 1),
        &channel_0_partition_indptr_ext,
        &err));
    OCL_CHECK(err, channel_0_indices_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(packed_index_t) * channel_0_indices.size(),
        &channel_0_indices_ext,
        &err));
    OCL_CHECK(err, channel_1_partition_indptr_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(unsigned int) * (num_col_partitions + 1),
        &channel_1_partition_indptr_ext,
        &err));
    OCL_CHECK(err, channel_1_indices_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(packed_index_t) * channel_1_indices.size(),
        &channel_1_indices_ext,
        &err));
    OCL_CHECK(err, kernel_results_buf = cl::Buffer(context,
        CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(data_t) * num_rows,
        &kernel_results_ext,
        &err));

    OCL_CHECK(err, err = the_kernel.setArg(0, vector_buf));
    OCL_CHECK(err, err = the_kernel.setArg(1, channel_0_partition_indptr_buf));
    OCL_CHECK(err, err = the_kernel.setArg(2, channel_0_indices_buf));
    OCL_CHECK(err, err = the_kernel.setArg(3, channel_1_partition_indptr_buf));
    OCL_CHECK(err, err = the_kernel.setArg(4, channel_1_indices_buf));
    OCL_CHECK(err, err = the_kernel.setArg(5, kernel_results_buf));
    OCL_CHECK(err, err = the_kernel.setArg(6, num_rows));
    OCL_CHECK(err, err = the_kernel.setArg(7, num_cols));
    OCL_CHECK(err, err = the_kernel.setArg(8, num_times));

    // Copy input data to Device Global Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({vector_buf,
                                                     channel_0_partition_indptr_buf,
                                                     channel_0_indices_buf,
                                                     channel_1_partition_indptr_buf,
                                                     channel_1_indices_buf}, 0 /* 0 means from host*/));
    q.finish();

    auto kernel_start = std::chrono::high_resolution_clock::now();

    // Invoking the kernel
    OCL_CHECK(err, err = q.enqueueTask(the_kernel));
    q.finish();

    auto kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
    double kernel_time_in_sec = kernel_time.count();
    std::cout << "kernel_time_in_sec = " << kernel_time_in_sec << std::endl;

    // Copy result from device to host
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({kernel_results_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

    // Calculate the throughput
    // TODO: do we want to count the padded marker?
    double throughput = num_times * (nnz * sizeof(unsigned int)); // indices
    throughput /= 1000;               // to KB
    throughput /= 1000;               // to MB
    throughput /= 1000;               // to GB
    throughput /= kernel_time_in_sec; // to GB/s
    std::cout << "Memory THROUGHPUT = " << throughput << " GB/s" << std::endl;

    std::cout << "Compute THROUGHPUT = " << throughput / sizeof(unsigned int)
              << " GOPS" << std::endl;

    bool match = verify(reference_results, kernel_results, num_rows);
    std::cout << (match ? "TEST PASSED" : "TEST FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}

#pragma GCC diagnostic pop
