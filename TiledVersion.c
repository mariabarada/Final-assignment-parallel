%%cu
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 32
#define ROWS_A 1000
#define COLS_A 900
#define ROWS_B 900
#define COLS_B 1000

__global__ void matrixMultiplication(float *A, float *B, float *C, int rows_A, int cols_A, int cols_B) {
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (cols_A + TILE_WIDTH - 1) / TILE_WIDTH; tile++) {
        if (tile * TILE_WIDTH + threadIdx.x < cols_A && row < rows_A) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * cols_A + tile * TILE_WIDTH + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (tile * TILE_WIDTH + threadIdx.y < cols_A && col < cols_B) {
            shared_B[threadIdx.y][threadIdx.x] = B[(tile * TILE_WIDTH + threadIdx.y) * cols_B + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rows_A && col < cols_B) {
        C[row * cols_B + col] = sum;
    }
}

void matrixMultiplicationSequential(float *A, float *B, float *C, int rows_A, int cols_A, int cols_B) {
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            float sum = 0.0f;
            for (int k = 0; k < cols_A; k++) {
                sum += A[i * cols_A + k] * B[k * cols_B + j];
            }
            C[i * cols_B + j] = sum;
        }
    }
}

void compareResults(float *C, float *C_seq, int size) {
    bool resultMatch = true;
    int mismatchIndex = -1;
    for (int i = 0; i < size; i++) {
        if (fabs(C[i] - C_seq[i]) > 1e-3) {
            resultMatch = false;
            mismatchIndex = i;
            break;
        }
    }

    if (resultMatch) {
        printf("Result matches the sequential calculation.\n");
    } else {
        printf("Result does not match the sequential calculation.\n");
        printf("Mismatch at index %d\n", mismatchIndex);
        printf("GPU Result: %f\n", C[mismatchIndex]);
        printf("Sequential Result: %f\n", C_seq[mismatchIndex]);
    }
}


int main() {
    int rows_A = ROWS_A;
    int cols_A = COLS_A;
    int rows_B = ROWS_B;
    int cols_B = COLS_B;

    // Check if the matrices can be multiplied
    if (cols_A != rows_B) {
        printf("Error: Invalid matrix dimensions for multiplication.\n");
        return 1;
    }

    // Allocate memory for matrices A, B, and C on the host
    float *h_A = (float *)malloc(rows_A * cols_A * sizeof(float));
    float *h_B = (float *)malloc(rows_B * cols_B * sizeof(float));
    float *h_C = (float *)malloc(rows_A * cols_B * sizeof(float));
    float *h_C_seq = (float *)malloc(rows_A * cols_B * sizeof(float));

    // Initialize matrices A and B with random values
    for (int i = 0; i < rows_A * cols_A; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < rows_B * cols_B; i++) {
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate memory for matrices A, B, and C on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, rows_A * cols_A * sizeof(float));
    cudaMalloc((void **)&d_B, rows_B * cols_B * sizeof(float));
    cudaMalloc((void **)&d_C, rows_A * cols_B * sizeof(float));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, rows_A * cols_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, rows_B * cols_B * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((cols_B + blockSize.x - 1) / blockSize.x, (rows_A + blockSize.y - 1) / blockSize.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
     cudaEventRecord(start);

    // Launch the matrix multiplication kernel on the device
    matrixMultiplication<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows_A, cols_A, cols_B);

            // Record the stop event
    cudaEventRecord(stop);

    // Synchronize to wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);



    // Copy the result matrix C from device to host
    cudaMemcpy(h_C, d_C, rows_A * cols_B * sizeof(float), cudaMemcpyDeviceToHost);
 printf("GPU Runtime: %.3f ms\n", milliseconds);
    // Perform sequential matrix multiplication on the host
    matrixMultiplicationSequential(h_A, h_B, h_C_seq, rows_A, cols_A, cols_B);

    // Compare the GPU and sequential results
    compareResults(h_C, h_C_seq, rows_A * cols_B);

    // Free memory on host and device
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_seq);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
