#ifndef GRAPHLILY_HW_APPLY_H_
#define GRAPHLILY_HW_APPLY_H_

#include "ap_fixed.h"

// data types
const unsigned PACK_SIZE = 8;
typedef unsigned IDX_T;
typedef struct {IDX_T data[PACK_SIZE];} PACKED_IDX_T;
// typedef ap_ufixed<32, 16, AP_RND, AP_SAT> VAL_T;
typedef float VAL_T;
typedef struct {VAL_T data[PACK_SIZE];} PACKED_VAL_T;
typedef struct {IDX_T index; VAL_T val;} IDX_VAL_T;

// mask type
typedef char MASK_T;
#define NOMASK      0
#define WRITETOZERO 1
#define WRITETOONE  2

// Kernel configurations
const unsigned BATCH_SIZE = 128;

#endif  // GRAPHLILY_HW_APPLY_H_
