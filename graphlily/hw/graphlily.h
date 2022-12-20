#ifndef SEEPENS_H
#define SEEPENS_H

#include <ap_int.h>
#include <tapa.h>

constexpr int NUM_CH_SPARSE = 24;

constexpr int WINDOW_SIZE = 8192;
constexpr int DEP_DIST_LOAD_STORE = 10;
constexpr int X_PARTITION_FACTOR = 8;
constexpr int URAM_DEPTH = (48 / NUM_CH_SPARSE) * 4096; // 16 -> 12,288, 24 -> 8,192

using float_v16 = tapa::vec_t<float, 16>;

void Serpens(tapa::mmap<int> edge_list_ptr,
             tapa::mmaps<ap_uint<512>, NUM_CH_SPARSE> edge_list_ch,
             tapa::mmap<float_v16> vec_X,
             tapa::mmap<float_v16> vec_Y,
             tapa::mmap<float_v16> vec_MK,
             tapa::mmap<float_v16> vec_Y_out,
             const int NUM_ITE, const int NUM_A_LEN, const int M, const int K,
             const int alpha_u, const int beta_u,
             const int ewise_add_val, const int assign_val, const float semi_zero,
             const char semi_op, const char mask_type, const char mode);

//-------------------------------------------------------------------------
// GraphLily semiring and mask types
//-------------------------------------------------------------------------

// semiring
typedef char semiring_t;
#define MULADD 0
#define ANDOR  1
#define ADDMIN 2

const float FLOAT_INF = 0x7f000000;

const float MulAddZero = 0;
const float AndOrZero  = 0;
const float AddMinZero = FLOAT_INF;

// const float MulAddOne = 1;
// const float AndOrOne  = 1;
// const float AddMinOne = 0;

// mask type
typedef char mask_t;
#define NOMASK      0
#define WRITETOZERO 1
#define WRITETOONE  2

#define MODE_SPMV 1
#define MODE_EWISE_ADD 2
#define MODE_DENSE_ASSIGN 4

#endif  // SEEPENS_H
