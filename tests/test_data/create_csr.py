import numpy as np
import scipy.sparse

num_rows = 10
num_cols = 10
nnz_per_row = 1
nnz = nnz_per_row * num_rows

indptr = np.array([i * nnz_per_row for i in range(num_rows + 1)], dtype='uint32')
# indices = np.array([i * num_cols / nnz_per_row % num_cols for i in range(nnz)], dtype='uint32')
indices = np.array([i for i in range(nnz)], dtype='uint32')
data = np.ones(nnz)

M = scipy.sparse.csr_matrix((data, indices, indptr), shape=(num_rows, num_cols), dtype=np.float32)
# print(M.toarray())
scipy.sparse.save_npz("eye_10_csr_float32.npz", M)
