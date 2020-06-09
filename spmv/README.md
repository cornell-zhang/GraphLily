SpMV
====

This is an SpMV design.

`v1`: fixed sparse matrix size (number of columns = 2048); no dataflow

`v2:` fixed sparse matrix size (number of columns = 2048); dataflow optimization

`v3`: arbitrary sparse matrix size; dataflow optimization

Compile-time constants are defined in .h file

## COMMAND LINE ARGUMENTS
Once the environment has been configured, the application can be executed by
```
./host <kernel_spmv XCLBIN>
```
