const unsigned int VDATA_SIZE = 8;

const unsigned int STRIDE = 2;

const unsigned int IN_SIZE = 1 * 1024 * 1024;

typedef unsigned int data_t;

typedef struct vector_data_type {
    data_t data[VDATA_SIZE];
} v_data_t;
