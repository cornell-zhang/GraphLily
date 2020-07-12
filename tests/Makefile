COMMON_REPO = ../
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

BUILD_DIR = ./build

HOST_ARCH = x86

CXXFLAGS += -Wall -O3 -g -std=c++11
CXXFLAGS += -I$(ABS_COMMON_REPO)

LDFLAGS += -lrt -lstdc++

include $(ABS_COMMON_REPO)/common/includes/xcl2/xcl2.mk
CXXFLAGS += $(xcl2_CXXFLAGS)
LDFLAGS += $(xcl2_LDFLAGS)

include $(ABS_COMMON_REPO)/common/includes/opencl/opencl.mk
CXXFLAGS += $(opencl_CXXFLAGS)
LDFLAGS += $(opencl_LDFLAGS)

include $(ABS_COMMON_REPO)/common/includes/graph_partitioning/graph_partitioning.mk
CXXFLAGS += $(graph_partitioning_CXXFLAGS)
LDFLAGS += $(graph_partitioning_LDFLAGS)

test_graph_partitioning: test_graph_partitioning.cpp
	g++ $(CXXFLAGS) test_graph_partitioning.cpp -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv test_graph_partitioning $(BUILD_DIR)/
	$(BUILD_DIR)/test_graph_partitioning

test_module: test_module.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) test_module.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv test_module $(BUILD_DIR)/
	$(BUILD_DIR)/test_module

all: test_graph_partitioning test_module

clean:
	rm -r $(BUILD_DIR)
