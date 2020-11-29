#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"


#include "graphlily/synthesizer/overlay_synthesizer.h"
#include "graphlily/app/bfs.h"
#include "graphlily/app/pagerank.h"

#include <iostream>
#include <ap_fixed.h>
#include <gtest/gtest.h>


std::string target = "sw_emu";
uint32_t out_buf_len = 512 * graphlily::num_cycles_float_add;
uint32_t vec_buf_len = 512 * graphlily::num_cycles_float_add;


void clean_proj_folder() {
    std::string command = "rm -rf ./" + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
}


template<typename data_t>
void verify(std::vector<float, aligned_allocator<float>> &reference_results,
            std::vector<data_t, aligned_allocator<data_t>> &kernel_results) {
    ASSERT_EQ(reference_results.size(), kernel_results.size());
    float epsilon = 0.0001;
    for (size_t i = 0; i < reference_results.size(); i++) {
        ASSERT_TRUE(abs(float(kernel_results[i]) - reference_results[i]) < epsilon);
    }
}


TEST(SynthesizeOverlay, NULL) {
    graphlily::synthesizer::OverlaySynthesizer synthesizer(graphlily::num_hbm_channels,
                                                           out_buf_len,
                                                           vec_buf_len);
    synthesizer.set_target(target);
    synthesizer.synthesize();
}


TEST(BFS, PullPush) {
    graphlily::app::BFS bfs(graphlily::num_hbm_channels, out_buf_len, vec_buf_len);
    bfs.set_target(target);
    bfs.set_up_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    std::string csr_float_npz_path = "/work/shared/common/research/graphblas/"
                                     "data/sparse_matrix_graph/uniform_10K_10_csr_float32.npz";
    bool skip_empty_rows = true;
    bfs.load_and_format_matrix(csr_float_npz_path, skip_empty_rows);
    bfs.send_matrix_host_to_device();

    uint32_t source = 0;
    uint32_t num_iterations = 10;

    auto reference_results = bfs.compute_reference_results(source, num_iterations);

    // pull
    auto kernel_results = bfs.pull(source, num_iterations);
    verify<graphlily::val_t>(reference_results, kernel_results);

    // // push
    // kernel_results = bfs.push(source, num_iterations);
    // verify<graphlily::val_t>(reference_results, kernel_results);

    // // pull_push
    // kernel_results = bfs.pull_push(source, num_iterations);
    // verify<graphlily::val_t>(reference_results, kernel_results);
}


TEST(PageRank, Pull) {
    graphlily::app::PageRank pagerank(graphlily::num_hbm_channels, out_buf_len, vec_buf_len);
    pagerank.set_target(target);
    pagerank.set_up_runtime("./" + graphlily::proj_folder_name + "/build_dir." + target + "/fused.xclbin");

    std::string csr_float_npz_path = "/work/shared/common/research/graphblas/"
                                     "data/sparse_matrix_graph/uniform_10K_10_csr_float32.npz";
    float damping = 0.9;
    bool skip_empty_rows = true;
    pagerank.load_and_format_matrix(csr_float_npz_path, damping, skip_empty_rows);
    pagerank.send_matrix_host_to_device();

    uint32_t num_iterations = 10;
    auto kernel_results = pagerank.pull(damping, num_iterations);
    auto reference_results = pagerank.compute_reference_results(damping, num_iterations);
    verify<graphlily::val_t>(reference_results, kernel_results);
}


TEST(CleanOverlay, NULL) {
    clean_proj_folder();
}


int main(int argc, char ** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#pragma GCC diagnostic pop
