make bench_bfs

num_channels=16
spmv_out_buf_len=1024000
spmspv_out_buf_len=256000
vec_buf_len=30720

bitstream=/work/shared/common/project_build/graphblas/bitstreams/
bitstream+=open_source_166MHz/overlay.xclbin

DATASET_PATH=/work/shared/common/project_build/graphblas/data/sparse_matrix_graph

DATSETS=(gplus_108K_13M_csr_float32.npz
         ogbl_ppa_576K_42M_csr_float32.npz
         hollywood_1M_113M_csr_float32.npz
         pokec_1633K_31M_csr_float32.npz
         ogbn_products_2M_124M_csr_float32.npz
         orkut_3M_213M_csr_float32.npz)

NUM_ITER=(7 11 10 11 23 6)

BUILD_DIR=./build

for ((i = 0; i < ${#DATSETS[@]}; i++)) do
    echo ${BUILD_DIR}/bench_bfs ${DATSETS[i]}
    ${BUILD_DIR}/bench_bfs $num_channels $spmv_out_buf_len $spmspv_out_buf_len $vec_buf_len \
        $bitstream $DATASET_PATH/${DATSETS[i]} ${NUM_ITER[i]}
done
