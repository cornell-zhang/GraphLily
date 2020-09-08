#ifndef __GRAPHBLAS_GLOBAL_H
#define __GRAPHBLAS_GLOBAL_H

#include <string>
#include <cstdlib>
#include <type_traits>

#include "ap_fixed.h"
#include "xcl2.hpp"


namespace {

std::string get_root_path() {
    char* root_path = getenv("GRAPHBLAS_ROOT_PATH");
    return root_path == NULL ? std::string("") : std::string(root_path);
}

} // anonymous namespace


namespace graphblas {

// The root path
const std::string root_path = get_root_path();

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
    std::cout << "Failed to find "  << device_name << ", exit!\n";
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

const int DDR[2] = {CHANNEL_NAME(32), CHANNEL_NAME(33)};

// Data type
template<typename dtype>
std::string dtype_to_str() {
    std::string str;
    if (std::is_same<dtype, unsigned int>::value) {
        str = "unsigned int";
    } else if (std::is_same<dtype, int>::value) {
        str = "int";
    } else if (std::is_same<dtype, float>::value) {
        str = "float";
    } else if (std::is_same<dtype, ap_ufixed<32, 1>>::value) {
        str = "ap_ufixed<32, 1>";
    } else {
        std::cerr << "Data type is not supported" << std::endl;
        exit(EXIT_FAILURE);
    }
    return str;
}

// Semiring type
enum SemiRingType {
    kMulAdd = 0,
    kLogicalAndOr = 1,
    kAddMin = 2,
};

// Mask type
enum MaskType {
    kNoMask = 0,
    kMaskWriteToZero = 1,
    kMaskWriteToOne = 2,
};

// Data types
typedef uint32_t index_t;
const uint32_t idx_marker = 0xffffffff;
const uint32_t pack_size = 16;
typedef struct {index_t data[pack_size];} packed_index_t;

typedef struct {index_t index; float val;} index_float_t;

using aligned_dense_float_vec_t = std::vector<float, aligned_allocator<float>>;
using aligned_sparse_float_vec_t = std::vector<index_float_t, aligned_allocator<index_float_t>>;

// Makefile for synthesizing xclbin
const std::string makefile_prologue =
    "COMMON_REPO = $(GRAPHBLAS_ROOT_PATH)\n"
    "\n"
    "DEVICE = /opt/xilinx/platforms/" + device_name + "/" + device_name + ".xpfm\n"
    "\n"
    "TEMP_DIR := ./_x.$(TARGET)\n"
    "BUILD_DIR := ./build_dir.$(TARGET)\n"
    "\n"
    "VPP := v++\n"
    "\n"
    "CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps\n"
    "\n"
    "FUSED_KERNEL = $(BUILD_DIR)/fused.xclbin\n"
    "\n"
    "emconfig.json:\n"
    "\temconfigutil --platform $(DEVICE)\n"
    "\n"
    "build: $(FUSED_KERNEL) emconfig.json\n"
    "\n";

const std::string makefile_epilogue =
    "$(FUSED_KERNEL): $(KERNEL_OBJS)\n"
    "\tmkdir -p $(BUILD_DIR)\n"
    "\t$(VPP) $(CLFLAGS) --temp_dir $(BUILD_DIR) -l $(LDCLFLAGS) -o'$@' $(+)\n";

std::string add_kernel_to_makefile(std::string kernel_name) {
    std::string makefile_body;
    makefile_body += ("LDCLFLAGS += --config " + kernel_name + ".ini" + "\n");
    makefile_body += ("KERNEL_OBJS += $(TEMP_DIR)/" + kernel_name + ".xo" + "\n");
    makefile_body += "\n";
    makefile_body += ("$(TEMP_DIR)/" + kernel_name + ".xo: " + kernel_name + ".cpp" + "\n");
    makefile_body += ("\tmkdir -p $(TEMP_DIR)\n");
    makefile_body += ("\t$(VPP) $(CLFLAGS) --temp_dir $(TEMP_DIR) -c -k " + kernel_name + " -I'$(<D)' -o'$@' '$<'\n");
    makefile_body += "\n";
    return makefile_body;
}

// Project folder name
const std::string proj_folder_name = "proj";

} // namespace graphblas

#endif // __GRAPHBLAS_GLOBAL_H
