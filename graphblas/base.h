#ifndef __GRAPHBLAS_BASE_H
#define __GRAPHBLAS_BASE_H

#include <string>

#include "xcl2.hpp"


namespace graphblas {

// The device
const std::string device_name = "xilinx_u280_xdma_201920_1";

// Find the device
cl::Device find_device() {
    auto devices = xcl::get_xil_devices();
    for (size_t i = 0; i < devices.size(); i++) {
        cl::Device device = devices[i];
        if (device.getInfo<CL_DEVICE_NAME>() == device_name) {
            return device;
        }
    }
    std::cout << "Failed to find "  << device_name << " , exit!\n";
    exit(EXIT_FAILURE);
}

// HBM channels
#define MAX_HBM_CHANNEL_COUNT 32
#define CHANNEL_NAME(n) n | XCL_MEM_TOPOLOGY
const int HBM[MAX_HBM_CHANNEL_COUNT] = {
    CHANNEL_NAME(0),  CHANNEL_NAME(1),  CHANNEL_NAME(2),  CHANNEL_NAME(3),  CHANNEL_NAME(4),
    CHANNEL_NAME(5),  CHANNEL_NAME(6),  CHANNEL_NAME(7),  CHANNEL_NAME(8),  CHANNEL_NAME(9),
    CHANNEL_NAME(10), CHANNEL_NAME(11), CHANNEL_NAME(12), CHANNEL_NAME(13), CHANNEL_NAME(14),
    CHANNEL_NAME(15), CHANNEL_NAME(16), CHANNEL_NAME(17), CHANNEL_NAME(18), CHANNEL_NAME(19),
    CHANNEL_NAME(20), CHANNEL_NAME(21), CHANNEL_NAME(22), CHANNEL_NAME(23), CHANNEL_NAME(24),
    CHANNEL_NAME(25), CHANNEL_NAME(26), CHANNEL_NAME(27), CHANNEL_NAME(28), CHANNEL_NAME(29),
    CHANNEL_NAME(30), CHANNEL_NAME(31)};

// Data type
enum DataType {
    kBoolean = 0,
    kUint = 1,
    kInt = 2,
    kFloat = 3,
    kUfixed_32_1 = 4,
};

// Semiring type
enum SemiRingType {
    kMulAdd = 0,
    kLogicalAndOr = 1,
    kTropical = 2, // TODO: find the appropriate name
};

// Index type and packed index type
typedef uint32_t index_t;
const uint32_t idx_marker = 0xffffffff;
const uint32_t pack_size = 16;
using packed_index_t = struct {index_t data[pack_size];};

} // namespace graphblas

#endif // __GRAPHBLAS_BASE_H
