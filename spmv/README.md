SpMV
====

This is an SpMV design.

`v1`: fixed sparse matrix size (number of columns = 2048); no dataflow

`v2:` fixed sparse matrix size (number of columns = 2048); dataflow optimization

`v3`: column partitioning to support arbitrary number of columns; dataflow optimization

`v4`: data type is ap_ufixed<32, 1>; column partitioning to support arbitrary number of columns; dataflow optimization

Compile-time constants are defined in .h file

Pipeline rewind is not useful.

## TODO
1. Optimize the loading of the dense vector. Broadcast write?

2. BRAM has two read ports. Two PEs share one replica of the vector?

3. Implement row partitioning to fully support arbitrary matrix sizes.

## Command Line Arguments
Once the environment has been configured, the application can be executed by
```
./host <kernel_spmv XCLBIN>
```
