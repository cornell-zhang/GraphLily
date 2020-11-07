#include "ap_fixed.h"

#define UFINF 65536
#define IDX_MARKER 0xffffffff

const unsigned PACK_SIZE = 8;
const unsigned NUM_PORT_PER_BANK = 1;
const unsigned NUM_BANK_PER_HBM_CHANNEL = PACK_SIZE / NUM_PORT_PER_BANK;
const unsigned BANK_ID_NBITS = 3;
const unsigned BANK_ID_MASK = 7;

// data types
typedef unsigned IDX_T;
typedef struct {IDX_T data[PACK_SIZE];} PACKED_IDX_T;
typedef ap_ufixed<32, 16, AP_RND, AP_SAT> VAL_T;
typedef struct {VAL_T data[PACK_SIZE];} PACKED_VAL_T;

typedef struct matrix_packet {
   PACKED_IDX_T indices;
   PACKED_VAL_T vals;
} MAT_PKT_T;

// semiring
typedef char OP_T;
#define MULADD 0
#define ANDOR  1
#define ADDMIN 2

const VAL_T MulAddZero = 0;
const VAL_T AndOrZero  = 0;
const VAL_T AddMinZero = UFINF;

const VAL_T MulAddOne = 1;
const VAL_T AndOrOne  = 1;
const VAL_T AddMinOne = 0;

typedef struct shuffle_1_inout_value_type {
    IDX_T row_idx;
    VAL_T mat_val;
} SF_1_IO_VAL_T;

typedef struct shuffle_1_inout_type {
    IDX_T index;
    SF_1_IO_VAL_T data;
} SF_1_IO_T;

typedef struct shuffle_2_inout_value_type {
    VAL_T vec_val;
    VAL_T mat_val;
} SF_2_IO_VAL_T;

typedef struct shuffle_2_inout_type {
    IDX_T index;
    SF_2_IO_VAL_T data;
} SF_2_IO_T;

// mask type
typedef char MASK_T;
#define NOMASK      0
#define WRITETOZERO 1
#define WRITETOONE  2

// Below kernel configurations will be overwritten by the compiler
// const unsigned OUT_BUF_LEN =;
// const unsigned VEC_BUF_LEN =;
// const unsigned NUM_HBM_CHANNEL =;
// const unsigned NUM_PE_TOTAL = PACK_SIZE * NUM_HBM_CHANNEL;
