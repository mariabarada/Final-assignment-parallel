%%cu
#include <stdio.h>
#include <stdlib.h>
#define ROWS_A 1100
#define COLS_A 1000
#define ROWS_B 1000
#define COLS_B 1100

// CUDA kernel to perform matrix multiplication
__global__ void matrixMultiplication(float *A, float *B, float *C, int rows_A, int cols_A, int cols_B) {
    // Calculate the global thread index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if the thread is within the matrix dimensions
    if (row < rows_A && col < cols_B) {
        float value = 0.0f;
        for (int k = 0; k < cols_A; k++) {
            value += A[row * cols_A + k] * B[k * cols_B + col];
        }
        C[row * cols_B + col] = value;
    }
}

int main() {

  int rows_A = ROWS_A;
    int cols_A = COLS_A;
    int rows_B = ROWS_B;
    int cols_B = COLS_B;
    // Allocate memory for the matrices on the host
    float *h_A = (float *)malloc(rows_A * cols_A * sizeof(float));
    float *h_B = (float *)malloc(rows_B * cols_B * sizeof(float));
    float *h_C = (float *)malloc(rows_A * cols_B * sizeof(float));
    
    // Initialize the matrices with random values
    for (int i = 0; i < rows_A * cols_A; i++) {
        h_A[i] = (float)(rand() % 100);
    }
    for (int i = 0; i < rows_B * cols_B; i++) {
        h_B[i] = (float)(rand() % 100);
    }
    
    // Allocate memory for the matrices on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, rows_A * cols_A * sizeof(float));
    cudaMalloc((void **)&d_B, rows_B * cols_B * sizeof(float));
    cudaMalloc((void **)&d_C, rows_A * cols_B * sizeof(float));
    
    // Copy the input matrices from the host to the device
    cudaMemcpy(d_A, h_A, rows_A * cols_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, rows_B * cols_B * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set the block and grid dimensions
    dim3 blockDim(32, 32);
    dim3 gridDim((cols_B + blockDim.x - 1) / blockDim.x, (rows_A + blockDim.y - 1) / blockDim.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
     cudaEventRecord(start);
    // Launch the matrix multiplication kernel
    matrixMultiplication<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows_A, cols_A, cols_B);
        // Record the stop event
    cudaEventRecord(stop);

    // Synchronize to wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // Copy the result matrix from the device to the host
    cudaMemcpy(h_C, d_C, rows_A * cols_B * sizeof(float), cudaMemcpyDeviceToHost);
 printf("GPU Runtime: %.3f ms\n", milliseconds);
   
    // Allocate memory for the result matrix on the host
float *h_C_seq = (float *)malloc(rows_A * cols_B * sizeof(float));

// Perform the sequential matrix multiplication on the host
for (int i = 0; i < rows_A; i++) {
    for (int j = 0; j < cols_B; j++) {
        float sum = 0.0f;
        for (int k = 0; k < cols_A; k++) {
            sum += h_A[i * cols_A + k] * h_B[k * cols_B + j];
        }
        h_C_seq[i * cols_B + j] = sum;
    }
}

// Compare the result matrices element-wise
bool resultMatch = true;
for (int i = 0; i < rows_A * cols_B; i++) {
    if (fabs(h_C[i] - h_C_seq[i]) > 1e-5) {
        resultMatch = false;
        break;
    }
}

// Print the result of the comparison
if (resultMatch) {
    printf("Result matches the sequential calculation.\n");
} else {
    printf("Result does not match the sequential calculation.\n");
}

// Free memory for the sequential result matrix
free(h_C_seq);
    // Free memory on the host and the device
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}