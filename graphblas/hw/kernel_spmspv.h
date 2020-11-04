#include "ap_fixed.h"

#define UFINF 65535

const unsigned PACK_SIZE = 8;
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

typedef struct vector_packet {
   IDX_T index;
   VAL_T   val;
} VEC_PKT_T;

// semiring definition
typedef char OP_T;
#define MULADD 0
#define ANDOR  1
#define ADDMIN 2

// mask type
typedef char MASK_T;
#define NOMASK      0
#define WRITETOZERO 1
#define WRITETOONE  2

const VAL_T MulAddZero = 0;
const VAL_T AndOrZero  = 0;
const VAL_T AddMinZero = UFINF;

const VAL_T MulAddOne = 1;
const VAL_T AndOrOne  = 1;
const VAL_T AddMinOne = 0;

// TODO: overload "<<" operator for better line tracing support
typedef struct shuffle_inout_vale_type {
    VAL_T   mat_val;
    VAL_T   vec_val;
} SF_IO_VAL_T;

// TODO: overload "<<" operator for better line tracing support
typedef struct shuffle_inout_type {
    IDX_T          index;
    SF_IO_VAL_T    data;
} SF_IO_T;

typedef struct vector_loader_out_type {
    IDX_T current_column_id;
    VAL_T vector_value;
} VL_O_T;

// Below kernel configurations will be overwritten by the compiler
// const unsigned OUT_BUF_LEN =;
