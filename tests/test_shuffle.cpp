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

#define DEBUG_PRINT(...) \
std::cout << __VA_ARGS__ << std::endl << std::flush;

std::string target = "hw_emu";
uint32_t test_len = 8;
uint32_t num_in_lanes = 8;
uint32_t num_out_lanes = 8;
uint32_t addr_mask = num_out_lanes - 1;

typedef struct {
    unsigned index;
    unsigned uuid;
} PayloadT;

const unsigned INVALID_UUID = 0;

using StreamT = std::vector<PayloadT, aligned_allocator<PayloadT> >;

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
void compute_ref(StreamT *test_input_host,
                 StreamT *ref_result_host) {
    for (size_t OLid = 0; OLid < num_out_lanes; OLid++) {
        ref_result_host[OLid].clear();
    }
    for (size_t ILid = 0; ILid < num_in_lanes; ILid++) {
        for (size_t i = 0; i < test_input_host[ILid].size(); i++) {
            PayloadT pld = (test_input_host[ILid].data())[i];
            unsigned index = pld.index;
            ref_result_host[index & addr_mask].push_back(pld);
        }
    }
}

bool find_payload(StreamT a, bool* checkout, unsigned uuid) {
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].uuid == uuid && !checkout[i]) {
            checkout[i] = true;
            return true;
        }
    }
    return false;
}

bool and_reduction(bool* arr, unsigned size) {
    for (size_t i = 0; i < size; i++) {
        if (!arr[i]) { return false; }
    }
    return true;
}

bool stream_match_ooo(StreamT ref, StreamT src) {
    if (ref.size() != src.size()) { return false; }
    unsigned size = ref.size();
    bool checkout[size];
    for (size_t i = 0; i < size; i++) {
        checkout[i] = false;
    }

    for (size_t i = 0; i < size; i++) {
        bool found = find_payload(ref, checkout, src[i].uuid);
        if (!found) { return false; }
    }

    bool all_checkout = and_reduction(checkout, size);
    if (!all_checkout) { return false; }

    return true;
}

void verify(StreamT *ref_result_host,
            StreamT *result_kernel) {
    for (size_t OLid = 0; OLid < num_out_lanes; OLid++) {
        bool match = stream_match_ooo(ref_result_host[OLid], result_kernel[OLid]);
        ASSERT_TRUE(match);
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
    command = "cp " + graphlily::root_path + "/graphlily/hw/shuffle.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/graphlily/hw/util.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/tests/testbench/shuffle_tb.h"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);
    command = "cp " + graphlily::root_path + "/tests/testbench/shuffle_tb.cpp"
                    + " " + graphlily::proj_folder_name + "/";
    DISP_EXE_CMD(command);

    // put configuration into pe_tb.h
    std::ofstream header(graphlily::proj_folder_name + "/shuffle_tb.h", std::ios_base::app);
    header << "const unsigned IN_BUF_SIZE = " << test_len << ";" << std::endl;
    header << "const unsigned OUT_BUF_SIZE = " << num_in_lanes * test_len << ";" << std::endl;
    header << "#endif // GRAPHLILY_TEST_TESTBENCH_SHUFFLE_TB_H_" << std::endl;
    header.close();

    // generate pe_tb.ini
    std::ofstream ini(graphlily::proj_folder_name + "/shuffle_tb.ini");
    ini << "[connectivity]" << std::endl;
    for (size_t i = 0; i < num_in_lanes; i++) {
        ini << "sp=shuffle_tb_1.test_input_stream" << i << ":HBM[" << i << "]" << std::endl;
    }
    for (size_t i = 0; i < num_out_lanes; i++) {
        ini << "sp=shuffle_tb_1.test_output_stream" << i << ":HBM[" << i + num_in_lanes << "]" << std::endl;
    }
    ini << "sp=shuffle_tb_1.num_payloads_in:DDR[0]" << std::endl;
    ini << "sp=shuffle_tb_1.num_payloads_out:DDR[0]" << std::endl;
    ini.close();

    // generate makefile
    std::ofstream makefile(graphlily::proj_folder_name + "/makefile");
    std::string makefile_body;
    makefile_body += "LDCLFLAGS += --config shuffle_tb.ini\n";
    makefile_body += "KERNEL_OBJS += $(TEMP_DIR)/shuffle_tb.xo\n";
    makefile_body += "\n";
    makefile_body += "$(TEMP_DIR)/shuffle_tb.xo: shuffle_tb.cpp\n";
    makefile_body += "\tmkdir -p $(TEMP_DIR)\n";
    makefile_body += "\t$(VPP) $(CLFLAGS) --temp_dir $(TEMP_DIR) -c -k shuffle_tb -I'$(<D)' -o'$@' '$<'\n";
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
void _test_pe_cluster(StreamT *test_input_host, bool debug) {
    // debug printout
    if (debug) {
        for (size_t ILid = 0; ILid < num_in_lanes; ILid++) {
            std::cout << "Input Lane[" << std::setw(2) << ILid << "]:";
            for (size_t i = 0; i < test_input_host[ILid].size(); i++) {
                std::cout << "{" << (test_input_host[ILid].data())[i].index << " | "
                          << (test_input_host[ILid].data())[i].uuid << " }, ";
            }
            std::cout << std::endl;
        }
    }

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
        std::cout << "Failed to find " << graphlily::device_name << " exit!\n";
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
    OCL_CHECK(err, kernel = cl::Kernel(program, "shuffle_tb", &err));

    cl::CommandQueue command_queue;
    OCL_CHECK(err, command_queue = cl::CommandQueue(context,
                                                    device,
                                                    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
                                                    &err));

    // prepare space for results
    StreamT kernel_result_host[num_out_lanes];
    for (size_t i = 0; i < num_out_lanes; i++) {
        kernel_result_host[i].resize(num_in_lanes * test_len);
        std::fill(kernel_result_host[i].begin(), kernel_result_host[i].end(), (PayloadT){i, INVALID_UUID});
    }

    // allocate memory
    cl_mem_ext_ptr_t test_input_stream_ext[num_in_lanes];
    for (size_t i = 0; i < num_in_lanes; i++) {
        test_input_stream_ext[i].obj = test_input_host[i].data();
        test_input_stream_ext[i].param = 0;
        test_input_stream_ext[i].flags = graphlily::HBM[i];
    }

    cl_mem_ext_ptr_t test_output_stream_ext[num_out_lanes];
    for (size_t i = 0; i < num_out_lanes; i++) {
        test_output_stream_ext[i].obj = kernel_result_host[i].data();
        test_output_stream_ext[i].param = 0;
        test_output_stream_ext[i].flags = graphlily::HBM[i + num_in_lanes];
    }

    std::vector<unsigned> num_payloads_in_host(8, 0);
    for (size_t i = 0; i < num_in_lanes; i++) {
        num_payloads_in_host[i] = test_input_host[i].size();
    }
    std::vector<unsigned> num_payloads_out_host(8, 0);
    CL_CREATE_EXT_PTR(num_payloads_in_ext, num_payloads_in_host.data(), graphlily::DDR[0]);
    CL_CREATE_EXT_PTR(num_payloads_out_ext, num_payloads_out_host.data(), graphlily::DDR[0]);

    cl::Buffer test_input_stream_buf[num_in_lanes];
    cl::Buffer test_output_stream_buf[num_out_lanes];
    cl::Buffer num_payloads_in_buf;
    cl::Buffer num_payloads_out_buf;

    for (size_t i = 0; i < num_in_lanes; i++) {
        OCL_CHECK(err, test_input_stream_buf[i] = cl::Buffer(context,
            CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
            sizeof(PayloadT) * num_payloads_in_host[i],
            &(test_input_stream_ext[i]),
            &err));
    }

    for (size_t i = 0; i < num_out_lanes; i++) {
        OCL_CHECK(err, test_output_stream_buf[i] = cl::Buffer(context,
            CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
            sizeof(PayloadT) * num_in_lanes * test_len,
            &(test_output_stream_ext[i]),
            &err));
    }

    OCL_CHECK(err, num_payloads_in_buf = cl::Buffer(context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(unsigned) * num_in_lanes,
        &num_payloads_in_ext,
        &err));

    OCL_CHECK(err, num_payloads_out_buf = cl::Buffer(context,
        CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        sizeof(unsigned) * num_out_lanes,
        &num_payloads_out_ext,
        &err));

    // migrate data
    for (size_t i = 0; i < num_in_lanes; i++) {
        OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects(
            {test_input_stream_buf[i]}, 0 /* 0 means from host*/));
        command_queue.finish();
    }
    OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects(
        {num_payloads_in_buf}, 0 /* 0 means from host*/));
    command_queue.finish();

    // set arguments
    for (size_t i = 0; i < num_in_lanes; i++) {
        OCL_CHECK(err, err = kernel.setArg(i, test_input_stream_buf[i]));
    }
    for (size_t i = 0; i < num_out_lanes; i++) {
        OCL_CHECK(err, err = kernel.setArg(i + num_in_lanes, test_output_stream_buf[i]));
    }
    OCL_CHECK(err, err = kernel.setArg(num_in_lanes + num_out_lanes, num_payloads_in_buf));
    OCL_CHECK(err, err = kernel.setArg(num_in_lanes + num_out_lanes + 1, num_payloads_out_buf));

    // launch kernel
    DEBUG_PRINT("INFO: [Shuffle TB-host] running kernel: shuffle testbench...");
    OCL_CHECK(err, err = command_queue.enqueueTask(kernel));
    command_queue.finish();
    DEBUG_PRINT("INFO: [Shuffle TB-host] kernel finished");

    // collect results
    OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects(
        {num_payloads_out_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
    command_queue.finish();
    for (size_t i = 0; i < num_out_lanes; i++) {
        OCL_CHECK(err, err = command_queue.enqueueMigrateMemObjects(
            {test_output_stream_buf[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
        command_queue.finish();
        kernel_result_host[i].resize(num_payloads_out_host[i]);
    }

    // compute reference
    StreamT ref_result_host[num_out_lanes];
    compute_ref(test_input_host, ref_result_host);

    // debug printout
    if (debug) {
        for (size_t OLid = 0; OLid < num_in_lanes; OLid++) {
            std::cout << "Reference Output Lane[" << std::setw(2) << OLid << "]:";
            for (size_t i = 0; i < ref_result_host[OLid].size(); i++) {
                std::cout << "{" << (ref_result_host[OLid].data())[i].index << " | "
                          << (ref_result_host[OLid].data())[i].uuid << " }, ";
            }
            std::cout << std::endl;
        }
        for (size_t OLid = 0; OLid < num_in_lanes; OLid++) {
            std::cout << "Kernel Output Lane[" << std::setw(2) << OLid << "]:";
            for (size_t i = 0; i < kernel_result_host[OLid].size(); i++) {
                std::cout << "{" << (kernel_result_host[OLid].data())[i].index << " | "
                          << (kernel_result_host[OLid].data())[i].uuid << " }, ";
            }
            std::cout << std::endl;
        }
    }

    // verify
    verify(ref_result_host, kernel_result_host);
}

//--------------------------------------------------------------------------------------------------
// Test cases
// NOTE! If we run all tests cases all at once, there could be memory overflow problems.
// If a segmentation fault happens, try to decrease the number of test cases launched in a run.
//--------------------------------------------------------------------------------------------------

void gen_no_conflict(int shift, StreamT* input) {
    unsigned uuid_cnt = 0;
    for (size_t ILid = 0; ILid < num_in_lanes; ILid++) {
        input[ILid].clear();
    }
    for (size_t i = 0; i < test_len; i++) {
        for (size_t ILid = 0; ILid < num_in_lanes; ILid++) {
            PayloadT pld;
            pld.index = (ILid + shift) % num_in_lanes;
            pld.uuid = uuid_cnt + 1;
            input[ILid].push_back(pld);
            uuid_cnt++;
        }
    }
}

void gen_load_imbabance(int num_lanes_used, StreamT* input) {
    unsigned uuid_cnt = 0;
    for (size_t ILid = 0; ILid < num_in_lanes; ILid++) {
        input[ILid].clear();
    }
    for (size_t i = 0; i < test_len; i++) {
        for (size_t ILid = 0; ILid < num_lanes_used; ILid++) {
            PayloadT pld;
            pld.index = ILid;
            pld.uuid = uuid_cnt + 1;
            input[ILid].push_back(pld);
            uuid_cnt++;
        }
    }
    for (size_t ILid = num_lanes_used; ILid < num_in_lanes; ILid++) {
        PayloadT pld;
        pld.index = ILid;
        pld.uuid = uuid_cnt + 1;
        input[ILid].push_back(pld);
        uuid_cnt++;
    }
}

void gen_conflict(int level, StreamT* input) {
    unsigned uuid_cnt = 0;
    for (size_t ILid = 0; ILid < num_in_lanes; ILid++) {
        input[ILid].clear();
    }
    for (size_t i = 0; i < test_len; i++) {
        for (size_t ILid = 0; ILid < num_in_lanes; ILid++) {
            PayloadT pld;
            if (ILid < level) {
                pld.index = 0;
            } else {
                pld.index = ILid;
            }
            pld.uuid = uuid_cnt + 1;
            input[ILid].push_back(pld);
            uuid_cnt++;
        }
    }
}

void gen_random(StreamT* input) {
    unsigned uuid_cnt = 0;
    for (size_t ILid = 0; ILid < num_in_lanes; ILid++) {
        input[ILid].clear();
    }
    for (size_t i = 0; i < test_len; i++) {
        for (size_t ILid = 0; ILid < num_in_lanes; ILid++) {
            PayloadT pld;
            pld.index = rand() % (8);
            pld.uuid = uuid_cnt + 1;
            input[ILid].push_back(pld);
            uuid_cnt++;
        }
    }
}

TEST(Build, Synthesize) {
    synthesize_tb();
}

TEST(Shuffle, DirectNoConflict) {
    StreamT input[num_in_lanes];
    gen_no_conflict(0, input);
    _test_pe_cluster(input, false);
}

TEST(Shuffle, CrossNoConflict) {
    StreamT input[num_in_lanes];
    for (int shift = 1; shift < 8; shift++) {
        gen_no_conflict(shift, input);
        _test_pe_cluster(input, false);
    }
}

TEST(Shuffle, LoadImbalance) {
    StreamT input[num_in_lanes];
    for (int Nused = 1; Nused < 8; Nused++) {
        gen_load_imbabance(Nused, input);
        _test_pe_cluster(input, false);
    }
}

TEST(Shuffle, Conflict) {
    StreamT input[num_in_lanes];
    gen_conflict(7, input);
    _test_pe_cluster(input, true);
}

TEST(Shuffle, Random) {
    StreamT input[num_in_lanes];
    gen_random(input);
    _test_pe_cluster(input, true);
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
