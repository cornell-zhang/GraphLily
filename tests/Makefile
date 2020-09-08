BUILD_DIR = ./build

HOST_ARCH = x86

CXXFLAGS += -Wall -O3 -g -std=c++11
CXXFLAGS += -I$(GRAPHBLAS_ROOT_PATH)

LDFLAGS += -lrt -lstdc++

include $(GRAPHBLAS_ROOT_PATH)/xrt/includes/xcl2/xcl2.mk
CXXFLAGS += $(xcl2_CXXFLAGS)
LDFLAGS += $(xcl2_LDFLAGS)

include $(GRAPHBLAS_ROOT_PATH)/xrt/includes/opencl/opencl.mk
CXXFLAGS += $(opencl_CXXFLAGS)
LDFLAGS += $(opencl_LDFLAGS)

CXXFLAGS += -I/work/shared/common/research/graphblas/software/cnpy
LDFLAGS += -L/work/shared/common/research/graphblas/software/cnpy/build -lcnpy

test_io: test_io.cpp
	g++ $(CXXFLAGS) test_io.cpp -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv test_io $(BUILD_DIR)/
	cd $(BUILD_DIR); ./test_io

test_module: test_module.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) test_module.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv test_module $(BUILD_DIR)/
	cd $(BUILD_DIR); ./test_module

test_app: test_app.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) test_app.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv test_app $(BUILD_DIR)/
	cd $(BUILD_DIR); ./test_app

all: test_io test_module test_app

clean:
	rm -rf $(BUILD_DIR)
