make bench_spmv

num_channels=16
out_buf_len=512000
vec_buf_len=30720

bitstream=/work/shared/common/research/graphblas/bitstreams/
bitstream+=overlay_16c_500K_250K_30K_stream64_149MHz/overlay.xclbin

DATASET_PATH=/work/shared/common/research/graphblas/data/sparse_matrix_graph

DATSETS=(gplus_108K_13M_csr_float32.npz
         ogbl_ppa_576K_42M_csr_float32.npz
         hollywood_1M_113M_csr_float32.npz
         pokec_1633K_31M_csr_float32.npz
         ogbn_products_2M_124M_csr_float32.npz
         orkut_3M_213M_csr_float32.npz
         uniform_conflict_free_1M_64_csr_float32.npz
         uniform_conflict_free_1M_256_csr_float32.npz)

BUILD_DIR=./build

for dataset in ${DATSETS[@]} do
    echo ${BUILD_DIR}/bench_spmv $dataset
    ${BUILD_DIR}/bench_spmv $num_channels $out_buf_len $vec_buf_len $bitstream $DATASET_PATH/$dataset
done
