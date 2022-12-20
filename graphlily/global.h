#ifndef GRAPHLILY_GLOBAL_H_
#define GRAPHLILY_GLOBAL_H_

#include <string>
#include <cstdlib>

#include "ap_fixed.h"
#include "tapa.h"

using tapa::aligned_allocator;

namespace graphlily {

// Kernel configurations
const uint32_t pack_size = 8;
const uint32_t spmv_row_interleave_factor = 1;
const uint32_t num_hbm_channels = 16;

// Data types (please change this according to the kernel!)
// using val_t = ap_ufixed<32, 8, AP_RND, AP_SAT>;
using val_t = float;
typedef uint32_t idx_t;
const uint32_t idx_marker = 0xffffffff;
typedef struct {idx_t data[pack_size];} packed_idx_t;

typedef struct {idx_t index; val_t val;} idx_val_t;
typedef struct {idx_t index; float val;} idx_float_t;

using aligned_dense_vec_t = std::vector<val_t, aligned_allocator<val_t>>;
using aligned_sparse_vec_t = std::vector<idx_val_t, aligned_allocator<idx_val_t>>;

using aligned_dense_float_vec_t = std::vector<float, aligned_allocator<float>>;
using aligned_sparse_float_vec_t = std::vector<idx_float_t, aligned_allocator<idx_float_t>>;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

// const val_t UINT_INF = 0xffffffff;
// const val_t UFIXED_INF = 255;
const val_t FLOAT_INF = 0x7f000000;

// Operation type, named as k<opx><op+>
enum OperationType {
    kMulAdd = 0,
    kLogicalAndOr = 1,
    kAddMin = 2,
};

// Semiring definition
struct SemiringType {
    OperationType op;
    val_t one;  // identity element for operator <x> (a <x> one = a)
    val_t zero;  // identity element for operator <+> (a <+> zero = a)
};

const SemiringType ArithmeticSemiring = {kMulAdd, 1, 0};
const SemiringType LogicalSemiring = {kLogicalAndOr, 1, 0};
// const SemiringType TropicalSemiring = {kAddMin, 0, UINT_INF};
// const SemiringType TropicalSemiring = {kAddMin, 0, UFIXED_INF};
const SemiringType TropicalSemiring = {kAddMin, 0, FLOAT_INF};

// Mask type
enum MaskType {
    kNoMask = 0,
    kMaskWriteToZero = 1,
    kMaskWriteToOne = 2,
};

#define MODE_SPMV 1
#define MODE_EWISE_ADD 2
#define MODE_DENSE_ASSIGN 4

using float_v16 = tapa::vec_t<float, 16>;
constexpr int NUM_CH_SPARSE = 24;
constexpr int matrix_round_size = 1024;

#define frtPlaceholder(x) fpga::Placeholder((x).data(), (x).size())
#define frtReadWrite(x) fpga::ReadWrite((x).data(), (x).size())
#define frtReadOnlyToDevice(x) fpga::ReadOnly((x).data(), (x).size())
#define frtWriteOnlyToDevice(x) fpga::WriteOnly((x).data(), (x).size())

//------------------------------------------
// Kernel types
//------------------------------------------

// semiring
typedef char semiring_t;
#define MULADD 0
#define ANDOR  1
#define ADDMIN 2

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

//------------------------------------------
// Utilities
//------------------------------------------

// convert a sparse vector to dense
template<typename sparse_vec_t, typename dense_vec_t, typename val_t>
dense_vec_t convert_sparse_vec_to_dense_vec(const sparse_vec_t &sparse_vector,
                                            uint32_t range,
                                            val_t zero) {
    dense_vec_t dense_vector(range);
    std::fill(dense_vector.begin(), dense_vector.end(), zero);
    int nnz = sparse_vector[0].index;
    for (int i = 1; i < nnz + 1; i++) {
        dense_vector[sparse_vector[i].index] = sparse_vector[i].val;
    }
    return dense_vector;
}

// used to calculate BANK_ID_NBITS
unsigned log2(unsigned x) {
    switch (x) {
        case    1: return 0;
        case    2: return 1;
        case    4: return 2;
        case    8: return 3;
        case   16: return 4;
        default  : return 0;
    }
}

// TODO: add unsigned/float support, if val_t is not ap_fixed?

// Pack the raw bits of the arbitrary precision variable (e.g. ap_ufixed) to a
// *pseudo* uint32, mainly used to pass ap_[type] to OpenCL kernel as scalar.
// The unpacker is defined as `LOAD_RAW_BITS_FROM_UINT` in hw/libfpga/hisparse.h
template<typename val_t>
inline unsigned pack_raw_bits_to_uint(val_t val) {
    ap_uint<32> temp;
    temp(31,0) = val(31,0);
    return temp.to_uint();
}

// Pack the raw bits of uint32 to a pseudo val_t type (e.g. ap_ufixed/ap_uint).
// Note: in this way, the logical value of val_t is NOT equal to the original
// uint32 in most cases. For instance, packing uint32(256) to ap_ufixed<32, 8>,
// while 256 exceeds the logical range of ap_ufixed<32, 8>.
template<typename val_t>
inline val_t pack_uint_to_raw_bits(unsigned val) {
    val_t temp;
    temp(31,0) = ap_uint<32>(val)(31,0);
    return temp;
}

// explict specialization for `float` data type
template<>
inline float pack_uint_to_raw_bits(unsigned val) {
    return (float)val;
}

}  // namespace graphlily

#endif  // GRAPHLILY_GLOBAL_H_
