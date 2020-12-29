import numpy as np
import scipy.sparse

num_channels = 16
pack_size = 8

def calculate_degree_standard_deviation(nnz_each_row):
    return nnz_each_row.std()

def calculate_imbalance_factor(nnz_each_row):
    num_PEs = num_channels * pack_size
    nnz_each_PE = np.zeros(num_PEs)
    step = num_PEs
    for i in range(num_PEs):
        nnz_each_PE[i] = nnz_each_row[i::step].sum()
    return nnz_each_PE.max() / nnz_each_PE.mean()

path = "/work/shared/common/research/graphblas/data/sparse_matrix_graph/"
datasets = ["gplus_108K_13M_csr_float32.npz",
            "ogbl_ppa_576K_42M_csr_float32.npz",
            "hollywood_1M_113M_csr_float32.npz",
            "pokec_1633K_31M_csr_float32.npz",
            "ogbn_products_2M_124M_csr_float32.npz",
            "orkut_3M_213M_csr_float32.npz"]

if __name__ == "__main__":
    for dataset in datasets:
        csr_matrix = scipy.sparse.load_npz(path + dataset)
        nnz_each_row = csr_matrix.indptr[1::] - csr_matrix.indptr[:-1:]
        standard_deviation = calculate_degree_standard_deviation(nnz_each_row)
        average_degree = csr_matrix.nnz / csr_matrix.shape[0]
        normalized_standard_deviation = standard_deviation / average_degree
        print(dataset)
        print("standard_deviation: ", standard_deviation)
        print("normalized_standard_deviation: ", normalized_standard_deviation)
        print("imbalance_factor: ", calculate_imbalance_factor(nnz_each_row))
