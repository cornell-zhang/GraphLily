HOST_ARCH = x86

CXXFLAGS += -Wall -O3 -g -std=c++14
CXXFLAGS += -I$(GRAPHLILY_ROOT_PATH)
CXXFLAGS += -I$(HLS_INCLUDE)

LDFLAGS += -lrt -lstdc++

include $(GRAPHLILY_ROOT_PATH)/xrt/includes/xcl2/xcl2.mk
CXXFLAGS += $(xcl2_CXXFLAGS)
LDFLAGS += $(xcl2_LDFLAGS)

include $(GRAPHLILY_ROOT_PATH)/xrt/includes/opencl/opencl.mk
CXXFLAGS += $(opencl_CXXFLAGS)
LDFLAGS += $(opencl_LDFLAGS)

BUILD_DIR = ./build
synthesize: synthesize.cpp $(xcl2_SRCS)
	g++ $(CXXFLAGS) synthesize.cpp $(xcl2_SRCS) -o $@ $(LDFLAGS)
	mkdir -p $(BUILD_DIR)
	mv synthesize $(BUILD_DIR)/
	cd $(BUILD_DIR); ./synthesize
