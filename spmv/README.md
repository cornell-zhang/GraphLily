SpMV
====

This is an SpMV design.

`v1`: fixed sparse matrix size (number of columns = 2048); no dataflow

`v2:` fixed sparse matrix size (number of columns = 2048); dataflow optimization

`v3`: column partitioning to support arbitrary number of columns; dataflow optimization

`v4`: data type is ap_ufixed<32, 1>; improve the efficiency of loading the dense vector by data packing; column partitioning to support arbitrary number of columns; dataflow optimization

Compile-time constants are defined in .h file

Pipeline rewind is not useful.

## TODO
1. BRAM has two read ports. Two PEs share one replica of the vector?

2. Implement row partitioning to fully support arbitrary matrix sizes.

3. The multiple streams should be independent, but they are not, causing deadlock when the depth is not large enough.

## Command Line Arguments
Once the environment has been configured, the application can be executed by
```
./host <kernel_spmv XCLBIN>
```
