COMMON_REPO = ../
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

BUILD_DIR := ./build

include $(ABS_COMMON_REPO)/common/includes/graph_partitioning/graph_partitioning.mk
CXXFLAGS = $(graph_partitioning_CXXFLAGS) -Wall -O3 -g -std=c++11

test_graph_partitioning: test_graph_partitioning.cpp
	g++ $(CXXFLAGS) test_graph_partitioning.cpp -o $@
	mkdir -p $(BUILD_DIR)
	mv test_graph_partitioning $(BUILD_DIR)/
	$(BUILD_DIR)/test_graph_partitioning

all: test_graph_partitioning

clean:
	rm -r $(BUILD_DIR)
