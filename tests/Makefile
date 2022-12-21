BUILD_DIR = ./build

HOST_ARCH = x86

CXXFLAGS += -Wall -g -std=c++17
CXXFLAGS += -O3
CXXFLAGS += -I..

LDFLAGS += -lrt -lstdc++

CXXFLAGS += -I/work/shared/common/project_build/graphblas/software/cnpy
LDFLAGS += -L/work/shared/common/project_build/graphblas/software/cnpy/build -lcnpy
LDFLAGS += -lgtest # use pkg in TAPA environment
LDFLAGS += -pthread

# TAPA workflow
LDFLAGS += -ltapa -lfrt -lglog -lgflags -lOpenCL

test_io: test_io.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) test_io.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv test_io $(BUILD_DIR)/
	cd $(BUILD_DIR); ./test_io

test_module_spmv_spmspv: test_module_spmv_spmspv.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) test_module_spmv_spmspv.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv test_module_spmv_spmspv $(BUILD_DIR)/
	echo "Please run ./test_module_spmv_spmspv manually currently"

test_module_apply: test_module_apply.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) test_module_apply.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv test_module_apply $(BUILD_DIR)/
	echo "Please run ./test_module_apply manually currently"

test_app: test_app.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) test_app.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv test_app $(BUILD_DIR)/
	cd $(BUILD_DIR); ./test_app

test_pe_cluster: test_pe_cluster.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) test_pe_cluster.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv test_pe_cluster $(BUILD_DIR)/
	cd $(BUILD_DIR); ./test_pe_cluster

test_shuffle: test_shuffle.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) test_shuffle.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv test_shuffle $(BUILD_DIR)/
	cd $(BUILD_DIR); ./test_shuffle

all: test_io test_module_spmv_spmspv test_module_apply test_app test_pe_cluster

clean:
	rm -rf $(BUILD_DIR)
