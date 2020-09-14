make bench_bfs

num_channels=4
bitstream=${GRAPHBLAS_ROOT_PATH}/benchmark/bitstreams/bfs_4_channel_16_pack_64K_VecBuf_150MHz.xclbin

PATH=/work/shared/common/research/graphblas/data/sparse_matrix_graph
DATSETS=(gplus_108K_13M_csr_float32.npz
         reddit_233K_115M_csr_float32.npz
         pokec_1633K_31M_csr_float32.npz
         ogbn_proteins_132K_79M_csr_float32.npz
         ogbl_ppa_576K_42M_csr_float32.npz
         rMat_64K_64_csr_float32.npz
         rMat_64K_256_csr_float32.npz
         rMat_256K_64_csr_float32.npz
         rMat_256K_256_csr_float32.npz)

BUILD_DIR=./build

for dataset in ${DATSETS[@]}
do
    echo ${BUILD_DIR}/bench_bfs $dataset
    ${BUILD_DIR}/bench_bfs $num_channels $bitstream $PATH/$dataset
done
