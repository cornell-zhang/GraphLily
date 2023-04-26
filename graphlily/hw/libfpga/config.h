#ifndef GRAPHLILY_HW_CONFIG_H_
#define GRAPHLILY_HW_CONFIG_H_

// !! Below kernel configurations will be overwritten by the compiler. For more
// details, please see the implementation in graphlily::synthesizer.
const unsigned SPMV_OUT_BUF_LEN = 1024;
const unsigned SPMSPV_OUT_BUF_LEN = 512;
const unsigned SPMV_VEC_BUF_LEN = 256;
#define NUM_HBM_CHANNEL 16

#endif  // GRAPHLILY_HW_CONFIG_H_