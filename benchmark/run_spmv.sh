# make bench_spmv

# these configs are not used in the current Serpens GraphLily
num_channels=16
spmv_out_buf_len=1048576
vec_buf_len=32768

if [[ -z "${BITSTREAM}" ]]; then
    echo "Please set `BITSTREAM` variable before running $0"
    exit 255
else
    bitstream="${BITSTREAM}"
fi

DATASET_PATH=/work/shared/common/project_build/graphblas/data/sparse_matrix_graph

DATSETS=(gplus_108K_13M_csr_float32.npz
         ogbl_ppa_576K_42M_csr_float32.npz
         hollywood_1M_113M_csr_float32.npz
         pokec_1633K_31M_csr_float32.npz
         ogbn_products_2M_124M_csr_float32.npz
         mouse_gene_45K_29M_csr_float32.npz
         live_journal_5M_69M_csr_float32.npz
         usroads_csr_float32.npz
         #orkut_3M_213M_csr_float32.npz
         )

BUILD_DIR=./build

for ((i = 0; i < ${#DATSETS[@]}; i++)) do
    echo ${BUILD_DIR}/bench_spmv ${DATSETS[i]}
    ${BUILD_DIR}/bench_spmv $num_channels $spmv_out_buf_len $vec_buf_len $bitstream $DATASET_PATH/${DATSETS[i]}
done
