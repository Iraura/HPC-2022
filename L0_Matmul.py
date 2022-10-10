import numpy as np
import time
from numba import cuda
import matplotlib.pyplot as plt

matrix_size = 500

# Inicialisation of matrixes for CPU
cpu_matrix1 = np.random.randint(0, 10, (matrix_size, matrix_size))
cpu_matrix2 = np.random.randint(0, 10, (matrix_size, matrix_size))
cpu_matrix_res = np.zeros((matrix_size, matrix_size), dtype=int)

# Function of MatMul on CPU
def cpu_mat_mul(A, B, C):
    for i in range(matrix_size):
        for j in range(matrix_size):
            res = 0
            for k in range(matrix_size):
                res += A[i, k] * B[k, j]
            C[i, j] = res


def cpu_calc():
    print("CPU started its work...")
    start_time = time.time()
    cpu_mat_mul(cpu_matrix1, cpu_matrix2, cpu_matrix_res)
    print("%s seconds is time for calculation on CPU" % (time.time() - start_time))

if __name__ == "__main__":
    cpu_calc()

