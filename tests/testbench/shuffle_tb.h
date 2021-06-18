#ifndef GRAPHLILY_TEST_TESTBENCH_SHUFFLE_TB_H_
#define GRAPHLILY_TEST_TESTBENCH_SHUFFLE_TB_H_

// data types
typedef struct shuffle_inout_data_type {
    unsigned uuid;
    unsigned padding;
} SF_IO_DATA_T;

typedef struct shuffle_inout_type {
    unsigned index;
    SF_IO_DATA_T data;
} SF_IO_T;

typedef struct testbench_interfece_type {
    unsigned index;
    unsigned uuid;
} TB_IFC_T;

const unsigned INVALID_UUID = 0;

const unsigned NUM_IN_LANES = 8;
const unsigned NUM_OUT_LANES = 8;
const unsigned ADDR_MASK = 7;

// Below configurations will be overwritten by the compiler
// const unsigned IN_BUF_SIZE =
// const unsigned OUT_BUF_SIZE =
// #endif // GRAPHLILY_TEST_TESTBENCH_SHUFFLE_TB_H_
