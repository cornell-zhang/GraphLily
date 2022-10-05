GraphLily: A Graph Linear Algebra Overlay on HBM-Equipped FPGAs
===============================================================

GraphLily is the first FPGA overlay for graph processing.
GraphLily supports a rich set of graph algorithms by adopting the GraphBLAS programming interface, which formulates graph algorithms as sparse linear algebra kernels.
GraphLily effectively utilizes the high bandwidth of HBM to accelerate SpMV and SpMSpV, the two widely-used kernels in GraphBLAS, by co-designing the data layout and the accelerator architecture.
GraphLily further builds a middleware to provide runtime support, enabling users to easily port existing GraphBLAS programs from CPUs/GPUs to FPGAs.

For more information, refer to our [ICCAD'21 paper](https://www.csl.cornell.edu/~zhiruz/pdfs/graphlily-iccad2021.pdf).
```
@article{hu2021graphlily,
  title={GraphLily: Accelerating Graph Linear Algebra on HBM-Equipped FPGAs},
  author={Hu, Yuwei and Du, Yixiao and Ustun, Ecenur and Zhang, Zhiru},
  journal={International Conference On Computer Aided Design},
  year={2021}
}
```

And in this branch, we integrated 16-HBM HiSparse SpMV into GraphLily, and rewrote 1-DDR SpMSpV with HiSparse modules (e.g., PE, Shuffle). For the related work of HiSparse, please see https://github.com/cornell-zhang/HiSparse.

## Prerequisites
- FPGA Card: Xilinx Alveo U280
- Hardware Platform: [xilinx_u280_gen3x16_xdma_base_1](https://docs.xilinx.com/r/en-US/ug1120-alveo-platforms/U280-Gen3x16-XDMA-base_1-Platform)
- Vendor Tool: Xilinx Vitis 2022.1.1
- XRT: 2022.1

Note: the compatibility of the tools can be found in [this document](https://docs.xilinx.com/r/en-US/ug1120-alveo-platforms/Alveo-Platforms).

## Setup

### Clone the repo
```
git clone git@github.com:cornell-zhang/GraphLily.git
export GRAPHLILY_ROOT_PATH=/path/to/GraphLily
```

### Get the bitstream
To generate a new bitstream:
```bash
cd GraphLily/generate_bitstream
make synthesize
```

### Prepare datasets
The input is an adjacency matrix in csr format stored as a scipy npz file. Please install [cnpy](https://github.com/rogersce/cnpy), which is required for data loading.

Our ICCAD'21 paper evaluated the following six graph datasets:

- [googleplus](https://drive.google.com/file/d/1Wv9C7s0lK0KdrRPUsTqjlENvbMMKfykg/view?usp=sharing)
- [ogbl-ppa](https://drive.google.com/file/d/189Qp9h4BxXR8dAiQdmJWkW89y08eU5qR/view?usp=sharing)
- [hollywood](https://drive.google.com/file/d/1irBTVuYdJaMXQTUGQh7AerBjs784ykeO/view?usp=sharing)
- [pokec](https://drive.google.com/file/d/1UEwsIYgNWmm3ucBfatjg_lmG25oXWWI-/view?usp=sharing)
- [ogbn-products](https://drive.google.com/file/d/1yBJjW5aRpJt2if32gOWSmaYcI10KDQj0/view?usp=sharing)
- [orkut](https://drive.google.com/file/d/1Am0hPLhGNAwjYWt5nd_-XsIaKBiWcwqt/view?usp=sharing)

## Benchmark
Go to the GraphLily/benchmark folder, modify the cnpy path in Makefile, modify the bitstream path and the datasets path in run_bfs.sh, then run the script:
```bash
cd GraphLily/benchmark
bash run_bfs.sh
```

## Test
To do quick debug or test after tweaking the designs in GraphLily/graphlily, just go to the GraphLily/tests, build and run test programs (HW synthesis is included in those programs):
```bash
cd GraphLily/tests
make test_module_spmv_spmspv # generate bitstream and run module test by one command
```