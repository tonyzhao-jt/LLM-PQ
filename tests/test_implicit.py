import numpy as np
from scipy.signal import convolve2d

# Define the kernel and matrix of random numbers
conv_kernel = np.array([
    [1, -1],
    [0, 1]
])
N = 10
L = 12
rand = np.random.rand(N, L)

# Perform the convolution using convolve2d
result1 = convolve2d(rand, conv_kernel, mode='valid')
print(result1)

# Reshape the kernel and input matrix into 1D arrays
kernel_flat = conv_kernel.flatten()
rand_flat = rand.flatten()

# Compute the size of the output after convolution
M = (N - conv_kernel.shape[0] + 1) * (L - conv_kernel.shape[1] + 1)

# Define the convolution matrix
conv_matrix = np.zeros((M, N * L))

# Define the index of the convolution matrix
conv_row = 0

# Perform the convolution using matrix multiplication
for i in range(N - conv_kernel.shape[0] + 1):
    for j in range(L - conv_kernel.shape[1] + 1):
        # Select the submatrix of the input matrix for this convolution
        submatrix = rand[i:i+conv_kernel.shape[0], j:j+conv_kernel.shape[1]]
        
        # Flatten the submatrix into a 1D array
        submatrix_flat = submatrix.flatten()
        
        # Insert the flattened submatrix into the convolution matrix
        conv_matrix[conv_row, i*L+j:i*L+j+conv_kernel.size] = submatrix_flat * kernel_flat
        
        conv_row += 1

# Reshape the output of the convolution into a 2D matrix
result2 = conv_matrix.dot(rand_flat).reshape((N - conv_kernel.shape[0] + 1, L - conv_kernel.shape[1] + 1))

# Compare the outputs
print(np.allclose(result1, result2))