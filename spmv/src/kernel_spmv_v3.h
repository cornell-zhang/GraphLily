const unsigned int VDATA_SIZE = 16;
const unsigned int NUM_PE_PER_HBM_CHANNEL = VDATA_SIZE;
const unsigned int NUM_HBM_CHANNEL = 2;
const unsigned int NUM_PE_TOTAL = NUM_PE_PER_HBM_CHANNEL*NUM_HBM_CHANNEL;

const unsigned int MAX_NUM_ROWS = 100 * 1000;
const unsigned int VECTOR_BUFFER_LEN = 10 * 1000;

typedef int data_t;

typedef struct packed_data_type {
    data_t data[VDATA_SIZE];
} packed_data_t;

typedef struct packed_index_type {
    unsigned int data[VDATA_SIZE];
} packed_index_t;

typedef struct idx_val_pair_type {
    unsigned int idx;
    data_t val;
} idx_val_pair_t;

#define VAL_MARKER 0xffffffff
#define IDX_MARKER 0xffffffff
