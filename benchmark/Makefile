BUILD_DIR = ./build

HOST_ARCH = x86

CXXFLAGS += -Wall -O3 -g -std=c++17
CXXFLAGS += -I..

LDFLAGS += -lrt -lstdc++

CXXFLAGS += -I/work/shared/common/project_build/graphblas/software/cnpy
LDFLAGS += -L/work/shared/common/project_build/graphblas/software/cnpy/build -lcnpy

# TAPA workflow
LDFLAGS += -ltapa -lfrt -lglog -lgflags -lOpenCL

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

bench_sssp: bench_sssp.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) bench_sssp.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv bench_sssp $(BUILD_DIR)/

bench_pagerank: bench_pagerank.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) bench_pagerank.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv bench_pagerank $(BUILD_DIR)/

apps: bench_bfs bench_sssp bench_pagerank

clean:
	rm -rf $(BUILD_DIR)
