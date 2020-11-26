make bench_spmv

num_channels=8

bitstream=/work/shared/common/research/graphblas/bitstreams/
bitstream+=spmv_8_channel_8_pack_128K_VecBuf_320K_OutBuf_II_1_write_188MHz.xclbin

PATH=/work/shared/common/research/graphblas/data/sparse_matrix_graph

DATSETS=(gplus_108K_13M_csr_float32.npz
         reddit_233K_115M_csr_float32.npz
         com_youtube_1M_3M_csr_float32.npz
         web_google_876K_5M_csr_float32.npz
         live_journal_5M_69M_csr_float32.npz
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
    echo ${BUILD_DIR}/bench_spmv $dataset
    ${BUILD_DIR}/bench_spmv $num_channels $bitstream $PATH/$dataset
done
