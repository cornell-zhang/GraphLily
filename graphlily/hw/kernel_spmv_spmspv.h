#ifndef GRAPHLILY_HW_SPMV_SPMSPV_H_
#define GRAPHLILY_HW_SPMV_SPMSPV_H_

#include "ap_fixed.h"
#include "./math_constants.h"

#define IDX_MARKER 0xffffffff

const unsigned PACK_SIZE = 8;
const unsigned NUM_PORT_PER_BANK = 1;
const unsigned NUM_BANK_PER_HBM_CHANNEL = PACK_SIZE / NUM_PORT_PER_BANK;
const unsigned BANK_ID_NBITS = 3;
const unsigned BANK_ID_MASK = 7;

const unsigned NUM_CYCLES_FLOAT_ADD = 10;

// data types
typedef unsigned IDX_T;
typedef struct {IDX_T data[PACK_SIZE];} PACKED_IDX_T;
// typedef ap_ufixed<32, 16, AP_RND, AP_SAT> VAL_T;
typedef float VAL_T;
typedef struct {VAL_T data[PACK_SIZE];} PACKED_VAL_T;

typedef struct {
   PACKED_IDX_T indices;
   PACKED_VAL_T vals;
} SPMV_MAT_PKT_T;

typedef SPMV_MAT_PKT_T SPMSPV_MAT_PKT_T;

typedef struct {IDX_T index; VAL_T val;} IDX_VAL_T;

// semiring
typedef char OP_T;
#define MULADD 0
#define ANDOR  1
#define ADDMIN 2

const VAL_T MulAddZero = 0;
const VAL_T AndOrZero  = 0;
// const VAL_T AddMinZero = UFIXED_INF;
const VAL_T AddMinZero = FLOAT_INF;

const VAL_T MulAddOne = 1;
const VAL_T AndOrOne  = 1;
const VAL_T AddMinOne = 0;

// mask type
typedef char MASK_T;
#define NOMASK      0
#define WRITETOZERO 1
#define WRITETOONE  2

// Below kernel configurations will be overwritten by the compiler
// const unsigned OUT_BUF_LEN =;
// const unsigned VEC_BUF_LEN =;
// const unsigned NUM_HBM_CHANNEL =;
// #define NUM_HBM_CHANNEL x
// const unsigned SPMV_NUM_PE_TOTAL = PACK_SIZE * NUM_HBM_CHANNEL;

// #endif  // GRAPHLILY_HW_SPMV_SPMSPV_H_
