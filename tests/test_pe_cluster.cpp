#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <vector>
#include <ap_fixed.h>
#include <gtest/gtest.h>

#include "graphlily/global.h"

#define DISP_EXE_CMD(cmd)\
std::cout << cmd << std::endl;\
system(cmd.c_str());

#define CL_CREATE_EXT_PTR(name, data, channel)\
cl_mem_ext_ptr_t name;\
name.obj = data;\
name.param = 0;\
name.flags = channel;

std::string target = "sw_emu";
uint32_t test_input_len = 64;
uint32_t test_bank_size = 1024;
uint32_t num_PE = 8;
uint32_t bank_id_nbits = unsigned(std::log2(num_PE));

typedef struct {
    graphlily::val_t mat_val;
    graphlily::val_t vec_val;
} PayloadDataT;

typedef struct {
    graphlily::idx_t index;
    PayloadDataT data;
} PayloadT;

//--------------------------------------------------------------------------------------------------
// clean stuff
//--------------------------------------------------------------------------------------------------

void clean_proj_folder() {
    std::string command = "rm -rf ./" + graphlily::proj_folder_name;
    DISP_EXE_CMD(command);
}

//--------------------------------------------------------------------------------------------------
// reference and verify utils
//--------------------------------------------------------------------------------------------------
void compute_ref(std::vector<PayloadT> &test_input_host,
                 std::vector<graphlily::val_t> &ref_result_host) {
    // input buffer
    PayloadT input_buffer[num_PE][test_input_len];

    // output buffer
    graphlily::val_t output_buffer[num_PE][test_bank_size];

    // reset output buffer
    for (unsigned PEid = 0; PEid < num_PE; PEid++) {
        for (unsigned i = 0; i < test_bank_size; i++) {
            output_buffer[PEid][i] = 0;
        }
    }

    // initialize input buffer
    for (unsigned PEid = 0; PEid < num_PE; PEid++) {
        for (unsigned i = 0; i < test_input_len; i++) {
            input_buffer[PEid][i] = test_input_host[PEid * test_input_len + i];
        }
    }

    // compute
    for (unsigned PEid = 0; PEid < num_PE; PEid++) {
        for (unsigned i = 0; i < test_input_len; i++) {
            output_buffer[PEid][input_buffer[PEid][i].index >> bank_id_nbits]
                += input_buffer[PEid][i].data.mat_val * input_buffer[PEid][i].data.vec_val;
        }
    }

    // write back to results
    for (unsigned i = 0; i < test_bank_size; i++) {
        for (unsigned PEid = 0; PEid < num_PE; PEid++) {
            ref_result_host[i * num_PE + PEid] = output_buffer[PEid][i];
        }
    }
}

void verify(std::vector<graphlily::val_t> reference_results,
            std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> kernel_results) {
    float epsilon = 0.0001;
    ASSERT_EQ(reference_results.size(), kernel_results.size());
    for (size_t i = 0; i < test_bank_size; i++) {
        bool match = abs(float(kernel_results[i] - reference_results[i])) < epsilon;
        if (!match) {
            std::cout << "Error: Result mismatch"
                      << std::endl;
            std::cout << "i = " << i
                      << " Reference result = " << reference_results[i]
                      << " Kernel result = " << kernel_results[i]
                      << std::endl;
            ASSERT_TRUE(match);
        }
    }
}

//--------------------------------------------------------------------------------------------------
// synthesizer
//--------------------------------------------------------------------------------------------------
void synthesize_tb() {
    // create proj directory
    std::string command = "mkdir -p " + graphlily::proj_folder_name;
    DISP_EXE_CMD(command);

    // copy source code
    command = "cp " + graphlily::root_path + "/graphlily/hw/ufixed_pe_fwd.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/graphlily/hw/math_constants.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/graphlily/hw/util.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/tests/testbench/pe_tb.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/tests/testbench/pe_tb.cpp"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);

    // put configuration into pe_tb.h
    std::ofstream header(graphlily::proj_folder_name + "/pe_tb.h", std::ios_base::app);
    header << "const unsigned NUM_PE = " << num_PE << ";" << std::endl;
    header << "const unsigned BANK_ID_NBITS = " << bank_id_nbits << ";" << std::endl;
    header << "const unsigned BANK_SIZE = " << test_bank_size << ";" << std::endl;
    header << "const unsigned IN_BUF_SIZE = " << test_input_len << ";" << std::endl;
    header << "#endif // GRAPHLILY_TEST_TESTBENCH_PE_TB_H_" << std::endl;
    header.close();

    // generate pe_tb.ini
    std::ofstream ini(graphlily::proj_folder_name + "/pe_tb.ini");
    ini << "[connectivity]" << std::endl;
    ini << "sp=pe_tb_1.test_addr_gmem:DDR[0]" << std::endl;
    ini << "sp=pe_tb_1.test_mat_gmem:DDR[0]" << std::endl;
    ini << "sp=pe_tb_1.test_vec_gmem:DDR[0]" << std::endl;
    ini << "sp=pe_tb_1.result_gmem:DDR[1]" << std::endl;
    ini.close();

    // generate makefile
    std::ofstream makefile(graphlily::proj_folder_name + "/makefile");
    std::string makefile_body;
    makefile_body += "LDCLFLAGS += --config pe_tb.ini\n";
    makefile_body += "KERNEL_OBJS += $(TEMP_DIR)/pe_tb.xo\n";
    makefile_body += "\n";
    makefile_body += "$(TEMP_DIR)/pe_tb.xo: pe_tb.cpp\n";
    makefile_body += "\tmkdir -p $(TEMP_DIR)\n";
    makefile_body += "\t$(VPP) $(CLFLAGS) --temp_dir $(TEMP_DIR) -c -k pe_tb -I'$(<D)' -o'$@' '$<'\n";
    makefile_body += "\n";
    makefile << "TARGET := " << target << "\n" << std::endl;
    makefile << graphlily::makefile_prologue << makefile_body << graphlily::makefile_epilogue;
    makefile.close();

    // switch to build folder and build
    command = "cd " + graphlily::proj_folder_name + "; " + "make build";
    DISP_EXE_CMD(command);
    if (target == "sw_emu" || target == "hw_emu") {
        command = "cp " + graphlily::proj_folder_name + "/emconfig.json " + ".";
        DISP_EXE_CMD(command);
    }
}

//--------------------------------------------------------------------------------------------------
// test harness
//--------------------------------------------------------------------------------------------------
void _test_pe_cluster(std::vector<PayloadT> &test_input_host) {
    // set up runtime
    cl_int err;
    if (target == "sw_emu" || target == "hw_emu") {
        setenv("XCL_EMULATION_MODE", target.c_str(), true);
    }
    cl::Device device;
    bool found_device = false;
    auto devices = xcl::get_xil_devices();
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i].getInfo<CL_DEVICE_NAME>() == graphlily::device_name) {
            device = devices[i];
            found_device = true;
            break;
        }
    }
    if (!found_device) {
        std::cout << "Failed to find " << graphlily::device_name << ", exit!\n";
        exit(EXIT_FAILURE);
    }
    cl::Context context = cl::Context(device, NULL, NULL, NULL);
    auto file_buf = xcl::read_binary_file("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");
    cl::Program::Binaries binaries{{file_buf.data(), file_buf.size()}};
    cl::Program program(context, {device}, binaries, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file\n";
    } else {
        std::cout << "Successfully programmed device with xclbin file\n";
    }
    cl::Kernel kernel;
    OCL_CHECK(err, kernel = cl::Kernel(program, "pe_tb", &err));

    cl::CommandQueue command_queue;
    OCL_CHECK(err, command_queue = cl::CommandQueue(context,
                                                    device,
                                                    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
                                                    &err));

    // prepare space for results
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> kernel_result_host;
    kernel_result_host.resize(test_bank_size * num_PE);
    std::fill(kernel_result_host.begin(), kernel_result_host.end(), 0);

    // allocate memory
    std::vector<graphlily::idx_t, aligned_allocator<graphlily::idx_t>> test_addr_host;
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> test_mat_host;
    std::vector<graphlily::val_t, aligned_allocator<graphlily::val_t>> test_vec_host;
    test_addr_host.resize(test_input_host.size());
    test_mat_host.resize(test_input_host.size());
    test_vec_host.resize(test_input_host.size());
    for (size_t i = 0; i < test_input_host.size(); i++) {
        test_addr_host[i] = test_input_host[i].index;
        test_mat_host[i] = test_input_host[i].data.mat_val;
        test_vec_host[i] = test_input_host[i].data.vec_val;
    }

    CL_CREATE_EXT_PTR(test_addr_ext, test_addr_host.data(), graphlily::DDR[0]);
    CL_CREATE_EXT_PTR(test_mat_ext, test_mat_host.data(), graphlily::DDR[0]);
    CL_CREATE_EXT_PTR(test_vec_ext, test_vec_host.data(), graphlily::DDR[0]);
    CL_CREATE_EXT_PTR(kernel_result_ext, kernel_result_host.data(), graphlily::DDR[1]);

    cl::Buffer test_addr_buf;
    cl::Buffer test_mat_buf;
    cl::Buffer test_vec_buf;
    cl::Buffer kernel_result_buf;

    OCL_CHECK(err, test_addr_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphlily::idx_t) * test_input_host.size(),
        &test_addr_ext,
        &err));

    OCL_CHECK(err, test_mat_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphlily::val_t) * test_input_host.size(),
        &test_mat_ext,
        &err));

    OCL_CHECK(err, test_vec_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphlily::val_t) * test_input_host.size(),
        &test_vec_ext,
        &err));

    OCL_CHECK(err, kernel_result_buf = cl::Buffer(context,
        CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(graphlily::val_t) * num_PE * test_bank_size,
        &kernel_result_ext,
        &err));

    // migrate data
    OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects(
        {test_addr_buf, test_mat_buf, test_vec_buf}, 0 /* 0 means from host*/));
    command_queue.finish();

    // set arguments
    OCL_CHECK(err, err = kernel.setArg(0, test_addr_buf));
    OCL_CHECK(err, err = kernel.setArg(1, test_mat_buf));
    OCL_CHECK(err, err = kernel.setArg(2, test_vec_buf));
    OCL_CHECK(err, err = kernel.setArg(3, kernel_result_buf));

    // launch kernel
    OCL_CHECK(err, err = command_queue.enqueueTask(kernel));
    command_queue.finish();

    // collect results
    OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects({kernel_result_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
    command_queue.finish();

    // compute reference
    std::vector<graphlily::val_t> ref_result_host;
    ref_result_host.resize(test_bank_size * num_PE);
    std::fill(ref_result_host.begin(), ref_result_host.end(), 0);
    compute_ref(test_input_host, ref_result_host);

    // verify
    verify(ref_result_host, kernel_result_host);
}

//--------------------------------------------------------------------------------------------------
// Test cases
//--------------------------------------------------------------------------------------------------

// generate payloads with a certain dependence distance.
// distance < 0 means no dependency.
std::vector<PayloadT> testcase_gen(int distance) {
    std::vector<PayloadT> test_input_host;
    graphlily::idx_t bank_address[num_PE][test_input_len];
    graphlily::val_t mat_vals[num_PE][test_input_len];
    graphlily::val_t vec_vals[num_PE][test_input_len];

    for (size_t i = 0; i < num_PE; i++) {
        for (size_t j = 0; j < test_input_len; j++) {
            if (distance < 0) {
                bank_address[i][j] = j % test_bank_size;
            } else {
                bank_address[i][j] = j % distance;
            }
            mat_vals[i][j] = 1;
            vec_vals[i][j] = 1;
        }
    }

    test_input_host.resize(test_input_len * num_PE);
    for (size_t i = 0; i < num_PE; i++) {
        for (size_t j = 0; j < test_input_len; j++) {
            test_input_host[i * test_input_len + j].index = bank_address[i][j] << bank_id_nbits;
            test_input_host[i * test_input_len + j].data.mat_val = mat_vals[i][j];
            test_input_host[i * test_input_len + j].data.vec_val = vec_vals[i][j];
        }
    }
    return test_input_host;
}

TEST(Build, Synthesize) {
    synthesize_tb();
}

TEST(PeCluster, NoDep) {
    std::vector<PayloadT> input = testcase_gen(-1);
    _test_pe_cluster(input);
}

TEST(PeCluster, Dist1) {
    std::vector<PayloadT> input = testcase_gen(1);
    _test_pe_cluster(input);
}

TEST(PeCluster, Dist2) {
    std::vector<PayloadT> input = testcase_gen(2);
    _test_pe_cluster(input);
}

TEST(PeCluster, Dist3) {
    std::vector<PayloadT> input = testcase_gen(3);
    _test_pe_cluster(input);
}

TEST(PeCluster, Dist4) {
    std::vector<PayloadT> input = testcase_gen(4);
    _test_pe_cluster(input);
}

TEST(CleanUp, CleanProjDir) {
    clean_proj_folder();
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#pragma GCC diagnostic pop
