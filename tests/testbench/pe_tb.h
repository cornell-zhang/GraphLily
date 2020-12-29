#ifndef GRAPHLILY_TEST_TESTBENCH_PE_TB_H_
#define GRAPHLILY_TEST_TESTBENCH_PE_TB_H_

#include "ap_fixed.h"

#define MULADD 0
#define ANDOR  1
#define ADDMIN 2

// data types
typedef unsigned IDX_T;
typedef ap_ufixed<32, 8, AP_RND, AP_SAT> VAL_T;

typedef struct pe_input_val_type {
    VAL_T mat_val;
    VAL_T vec_val;
} PE_I_VAL_T;

typedef struct pe_input_type {
    IDX_T index;
    PE_I_VAL_T data;
} PE_I_T;

// Below configurations will be overwritten by the compiler
// const unsigned NUM_PE =
// const unsigned BANK_ID_NBITS =
// const unsigned BANK_SIZE =
// const unsigned IN_BUF_SIZE =
// #endif // GRAPHLILY_TEST_TESTBENCH_PE_TB_H_
