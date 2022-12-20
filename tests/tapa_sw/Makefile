PROJ_ROOT = ../..

CXX := g++

CXXFLAGS += -Wall -std=c++17
CXXFLAGS += -I./io -I$(PROJ_ROOT)/graphlily/hw
LDFLAGS += -lrt -lstdc++ -ltapa -lfrt -lglog -lgflags -lOpenCL -pthread

CXXFLAGS += -I$(CNPY_INCLUDE)
LDFLAGS += -L$(CNPY_LIB) -lcnpy

CXXFLAGS += -Wno-maybe-uninitialized
CXXFLAGS += -Wno-uninitialized
CXXFLAGS += -Wno-int-in-bool-context
CXXFLAGS += -Wno-unknown-pragmas
CXXFLAGS += -Wno-unused-function
CXXFLAGS += -Wno-unused-label
CXXFLAGS += -Wno-sign-compare

GRAPHLILY_HW := $(PROJ_ROOT)/graphlily/hw/graphlily.cpp

.PHONY: serpens_spmv assign_dense ewise_add all
serpens_spmv: serpens_spmv.cpp $(GRAPHLILY_HW)
	$(CXX) $(CXXFLAGS) -g $^ -o serpens_spmv $(LDFLAGS)

assign_dense: assign_dense.cpp $(GRAPHLILY_HW)
	$(CXX) $(CXXFLAGS) -g $^ -o assign_dense $(LDFLAGS)

ewise_add: ewise_add.cpp $(GRAPHLILY_HW)
	$(CXX) $(CXXFLAGS) -g $^ -o ewise_add $(LDFLAGS)

all: ewise_add assign_dense serpens_spmv

.PHONY: clean
clean:
	rm -rf .run *.log *.csv
	rm -rf serpens_spmv assign_dense ewise_add
