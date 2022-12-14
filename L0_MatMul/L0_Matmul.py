import numpy as np
import time
from numba import cuda
import math

matrix_size = 500

# Inicialisation of matrixes for CPU
cpu_matrix1 = np.random.randint(0, 10, (matrix_size, matrix_size))
cpu_matrix2 = np.random.randint(0, 10, (matrix_size, matrix_size))
cpu_matrix_res = np.zeros((matrix_size, matrix_size), dtype=int)

# Inicialisation of matrixes for GPU
gpu_matrix1 = cuda.to_device(cpu_matrix1)
gpu_matrix2 = cuda.to_device(cpu_matrix2)
gpu_matrix_res = cuda.device_array((len(cpu_matrix1), len(cpu_matrix2)))


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

@cuda.jit
def gpu_mat_mul(A, B, C):
    for i in range(matrix_size):
        for j in range(matrix_size):
            rez = 0
            for z in range(matrix_size):
                rez += A[i, z] * B[z, j]
            C[i, j] = rez

def gpu_calc():
    # Kernel parameters
    # Amount of threads in block
    threadsperblock = (32, 32)
    # Amount of blocks in grid by x
    blockspergrid_x = int(math.ceil(cpu_matrix1.shape[0] / threadsperblock[0]))
    # Amount of blocks in grid by y
    blockspergrid_y = int(math.ceil(cpu_matrix2.shape[1] / threadsperblock[1]))
    # Amount of blocks in whole grid
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print("Grid size = ", blockspergrid, threadsperblock)
    print("End of CPU work!\n")

    print("GPU started its work...")
    start_time = time.time()
    # Calculation on GPU on given blocks and threads
    gpu_mat_mul[blockspergrid, threadsperblock](gpu_matrix1, gpu_matrix2, gpu_matrix_res)
    print("%s seconds is time for calculation on GPU" % (time.time() - start_time))
    print("End of GPU work!\n")


if __name__ == "__main__":
    cpu_calc()
    gpu_calc()
    print("Hope this experiment went good!\n")

