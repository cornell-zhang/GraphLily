GraphLily: A Graph Linear Algebra Overlay on HBM-Equipped FPGAs
===============================================================

GraphLily is the **first** FPGA overlay for graph processing.

GraphLily supports a rich set of graph algorithms by adopting the [GraphBLAS](https://graphblas.org/) programming interface, which formulates graph algorithms as sparse linear algebra kernels.

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

Moreover, we made 3 new contributions as the follow-up works of ICCAD'21 GraphLily:

| Design | Mode | Description | Link |
|---|---|---|---|
| 16-HBM HiSparse SpMV + 1-DDR HiSparse SpMSpV | Pull/Pull-Push | First integration w/ the multi-HBM SpMV from FPGA'22 HiSparse work and the rewritten single-DDR SpMSpV. For more information about HiSparse, please see https://github.com/cornell-zhang/HiSparse | [hang_integration](https://github.com/cornell-zhang/GraphLily/tree/hang_integration) |
| 16-HBM HiSparse SpMV + 8-HBM HiSparse SpMSpV | Pull/Pull-Push | Second integration w/ the newly developed multi-HBM HiSparse SpMSpV | [hang_spmspv_hbm](https://github.com/cornell-zhang/GraphLily/tree/hang_spmspv_hbm) |
| 24-HBM Serpens SpMV | Pull | Third integration w/ [UCLA-VAST Serpens SpMV](https://github.com/UCLA-VAST/Serpens) (Note: this design is based on TAPA workflow) | [hang_serpens_spmv](https://github.com/cornell-zhang/GraphLily/tree/hang_serpens_spmv) |


## Prerequisites
- FPGA Card: Xilinx Alveo U280
- Hardware Platform: [xilinx_u280_gen3x16_xdma_base_1](https://docs.xilinx.com/r/en-US/ug1120-alveo-platforms/U280-Gen3x16-XDMA-base_1-Platform)
- Vendor Tool: Xilinx Vitis 2022.1.1
- XRT: 2022.1
- [googletest](https://github.com/google/googletest)
- [cnpy](https://github.com/rogersce/cnpy)

Note: the compatibility of the tools can be found in [this document](https://docs.xilinx.com/r/en-US/ug1120-alveo-platforms/Alveo-Platforms).

## Setup
__For internal developers__:
you can directly setup the environment by `source setupGraphLily.sh`, and jump to setp 3.

### 0. Install Google test and cnpy
We rely on Google test to manage our test cases.
Please install [googletest](https://github.com/google/googletest).
Please also install [cnpy](https://github.com/rogersce/cnpy), which is required for loading Numpy data from C++.
After installing these libiaries, please modify the corresponding paths in the Makefiles.

### 1. Clone the repo and setup the environment
```
git clone git@github.com:cornell-zhang/GraphLily.git
export GRAPHLILY_ROOT_PATH=/path/to/GraphLily
```
### 2. Prepare datasets
The input is an adjacency matrix in csr format stored as a scipy npz file.

Our ICCAD'21 paper evaluated the following six graph datasets:

- [googleplus](https://drive.google.com/file/d/1Wv9C7s0lK0KdrRPUsTqjlENvbMMKfykg/view?usp=sharing)
- [ogbl-ppa](https://drive.google.com/file/d/189Qp9h4BxXR8dAiQdmJWkW89y08eU5qR/view?usp=sharing)
- [hollywood](https://drive.google.com/file/d/1irBTVuYdJaMXQTUGQh7AerBjs784ykeO/view?usp=sharing)
- [pokec](https://drive.google.com/file/d/1UEwsIYgNWmm3ucBfatjg_lmG25oXWWI-/view?usp=sharing)
- [ogbn-products](https://drive.google.com/file/d/1yBJjW5aRpJt2if32gOWSmaYcI10KDQj0/view?usp=sharing)
- [orkut](https://drive.google.com/file/d/1Am0hPLhGNAwjYWt5nd_-XsIaKBiWcwqt/view?usp=sharing)

### 3. Test
To do quick debug or test after tweaking the designs in GraphLily/graphlily, just go to the GraphLily/tests, build and run test programs (HW synthesis is included in those programs).
```bash
cd GraphLily/tests
make test_module_spmv_spmspv # generate bitstream for testing and run module test by one command
```

### 4. Get the bitstream for benchmarking
To generate a new bitstream:
```bash
cd GraphLily/generate_bitstream
make synthesize
```

### 5. Benchmark
Go to the GraphLily/benchmark folder, modify the cnpy path in Makefile, modify the bitstream path and the datasets path in run_bfs.sh, then run the script:
```bash
cd GraphLily/benchmark
bash run_bfs.sh
```

### Troubleshooting

If you meet `undefined reference to cnpy::npz_load(std::string)` in link stage or `error while loading shared libraries: libcnpy.so: cannot open shared object file: No such file or directory` for execution, then check if cnpy is successfully installed and ensure the shared library can be found by dynamic linker/loader (e.g., adding the lib path to `LD_LIBRARY_PATH`). This also applies to the troubles related to googletest used in GraphLily.