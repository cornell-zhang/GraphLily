# TAPA Host for Unit Tests

Use `make all` to build the tests for 3 modules of GraphLily:

1. `serpens_spmv`: Serpens SpMV
2. `assign_dense`: Assign Dense Vector w/ Mask
3. `ewise_add`: Element-wise Scalar Add

Set the `DATASETS` variable to the path of graph datasets.

Set the `BITSTREAM` variable to the path of the bitstream (the target of which can be hw, Vitis hw_emu, or Vitis sw_emu). Otherwise, you can keep `BITSTREAM` empty or unset, if you want to run [TAPA sw_emu](https://tapa.readthedocs.io/en/release/getting_started.html#run-software-simulation).

Then simply run `./serpens_spmv` or other tests.

Note: the `io` directory here is slightly different from the `/path/to/graphlily/graphlily/io` directory literally, but is the same regrading functionality. We place this `io` directory here so that we can easily treat the combination of two directories (i.e., `tapa_sw` and `/path/to/graphlily/graphlily/hw`) as a standalone TAPA project.
