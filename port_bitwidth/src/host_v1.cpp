#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "xcl2.hpp"

#include "kernel_strided_access_v1.h"

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

    // Generate random input vectors
    std::vector<int, aligned_allocator<int>> in0(IN_SIZE);
    std::generate(in0.begin(), in0.end(), [&](){return std::rand() % 256;});

    // Initialize kernel_results to zero
    std::vector<int, aligned_allocator<int>> kernel_results(IN_SIZE);
    std::fill(kernel_results.begin(), kernel_results.end(), 0);

    // Compute reference_results
    std::vector<int, aligned_allocator<int>> reference_results(IN_SIZE);
    std::fill(reference_results.begin(), reference_results.end(), 0);
    for (size_t i = 0; i < IN_SIZE / VDATA_SIZE; i+=STRIDE) {
        for (size_t j = 0; j < VDATA_SIZE; j++) {
            reference_results[i*VDATA_SIZE + j] = in0[i*VDATA_SIZE + j] + 1;
        }
    }

    // Find the OpenCL binary file
    std::string binaryFile = argv[1];
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

    std::string kernel_name = "kernel_strided_access_v1";
    std::string kernel_name_full = kernel_name + ":{" + "kernel_strided_access_v1_" + "1" + "}";

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
            OCL_CHECK(err, the_kernel = cl::Kernel(program, kernel_name_full.c_str(), &err));
            valid_device++;
            break;  // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    unsigned int num_times = 1000;
    if (xcl::is_emulation()) {
        num_times = 1;
    }

    cl_mem_ext_ptr_t in0_ext;
    cl_mem_ext_ptr_t kernel_results_ext;

    cl::Buffer in0_buf;
    cl::Buffer kernel_results_buf;

    in0_ext.obj = in0.data();
    in0_ext.param = 0;
    in0_ext.flags = channels[0];

    kernel_results_ext.obj = kernel_results.data();
    kernel_results_ext.param = 0;
    kernel_results_ext.flags = channels[1];

    // Allocate memory on the FPGA
    OCL_CHECK(err, in0_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(data_t) * IN_SIZE,
        &in0_ext,
        &err));
    OCL_CHECK(err, kernel_results_buf = cl::Buffer(context,
        CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(data_t) * IN_SIZE,
        &kernel_results_ext,
        &err));

    OCL_CHECK(err, err = the_kernel.setArg(0, in0_buf));
    OCL_CHECK(err, err = the_kernel.setArg(1, kernel_results_buf));
    OCL_CHECK(err, err = the_kernel.setArg(2, num_times));

    // Copy input data to Device Global Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({in0_buf}, 0 /* 0 means from host*/));
    q.finish();
    std::cout << "after enqueueMigrateMemObjects" << std::endl;

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
    double throughput = num_times * (IN_SIZE / STRIDE * sizeof(data_t));
    throughput /= 1000;               // to KB
    throughput /= 1000;               // to MB
    throughput /= 1000;               // to GB
    throughput /= kernel_time_in_sec; // to GB/s
    std::cout << "THROUGHPUT = " << throughput << " GB/s" << std::endl;

    bool match = verify(reference_results, kernel_results, IN_SIZE);
    std::cout << (match ? "TEST PASSED" : "TEST FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
