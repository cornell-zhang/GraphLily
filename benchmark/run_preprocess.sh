export OMP_NUM_THREADS=16

make bench_preprocess

num_channels=16
spmv_out_buf_len=1048576
vec_buf_len=24576

DATASET_PATH=/work/shared/common/project_build/graphblas/data/

DATSETS=(sparse_matrix_graph/gplus_108K_13M_csr_float32.npz
         sparse_matrix_graph/ogbl_ppa_576K_42M_csr_float32.npz
         sparse_matrix_graph/hollywood_1M_113M_csr_float32.npz
         sparse_matrix_graph/pokec_1633K_31M_csr_float32.npz
         sparse_matrix_graph/ogbn_products_2M_124M_csr_float32.npz
         sparse_matrix_graph/mouse_gene_45K_29M_csr_float32.npz
         pruned_neural_network/transformer_50_33288_512_csr_float32.npz
         pruned_neural_network/transformer_60_33288_512_csr_float32.npz
         pruned_neural_network/transformer_70_33288_512_csr_float32.npz
         pruned_neural_network/transformer_80_33288_512_csr_float32.npz
         pruned_neural_network/transformer_90_33288_512_csr_float32.npz
         pruned_neural_network/transformer_95_33288_512_csr_float32.npz)

BUILD_DIR=./build

for ((i = 0; i < ${#DATSETS[@]}; i++)) do
    echo ${BUILD_DIR}/bench_preprocess ${DATSETS[i]}
    ${BUILD_DIR}/bench_preprocess $num_channels $spmv_out_buf_len $vec_buf_len $DATASET_PATH/${DATSETS[i]}
done
