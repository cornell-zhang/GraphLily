const unsigned int VDATA_SIZE = 16;
const unsigned int NUM_PE_PER_HBM_CHANNEL = VDATA_SIZE;
const unsigned int NUM_HBM_CHANNEL = 2;
const unsigned int NUM_PE_TOTAL = NUM_PE_PER_HBM_CHANNEL*NUM_HBM_CHANNEL;

const unsigned int NUM_ROWS = 2048;
const unsigned int NUM_COLS = 2028;

typedef unsigned int data_t;

typedef struct vector_data_type {
    data_t data[VDATA_SIZE];
} v_data_t;
