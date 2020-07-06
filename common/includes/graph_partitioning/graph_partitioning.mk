graph_partitioning_HDRS:=${COMMON_REPO}/common/includes/graph_partitioning/graph_partitioning.h
graph_partitioning_CXXFLAGS:=-I${COMMON_REPO}/common/includes/graph_partitioning -I/work/shared/common/research/graphblas/software/cnpy
graph_partitioning_LDFLAGS:=-L/work/shared/common/research/graphblas/software/cnpy/build -lcnpy
