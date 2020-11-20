HOST_ARCH = x86

CXXFLAGS += -Wall -O3 -g -std=c++11
CXXFLAGS += -I$(GRAPHLILY_ROOT_PATH)

LDFLAGS += -lrt -lstdc++

include $(GRAPHLILY_ROOT_PATH)/xrt/includes/xcl2/xcl2.mk
CXXFLAGS += $(xcl2_CXXFLAGS)
LDFLAGS += $(xcl2_LDFLAGS)

include $(GRAPHLILY_ROOT_PATH)/xrt/includes/opencl/opencl.mk
CXXFLAGS += $(opencl_CXXFLAGS)
LDFLAGS += $(opencl_LDFLAGS)

CXXFLAGS += -I/work/shared/common/research/graphblas/software/cnpy
LDFLAGS += -L/work/shared/common/research/graphblas/software/cnpy/build -lcnpy

BUILD_DIR = ./build

bench_spmv: bench_spmv.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) bench_spmv.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv bench_spmv $(BUILD_DIR)/

bench_spmspv: bench_spmspv.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) bench_spmspv.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv bench_spmspv $(BUILD_DIR)/

bench_bfs: bench_bfs.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) bench_bfs.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv bench_bfs $(BUILD_DIR)/

bench_pagerank: bench_pagerank.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) bench_pagerank.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv bench_pagerank $(BUILD_DIR)/

clean:
	rm -rf $(BUILD_DIR)
