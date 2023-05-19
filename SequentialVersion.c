#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Function declarations
void matrixMultiplicationSequential(float *A, float *B, float *C, int rows_A, int cols_A, int cols_B);

int main() {
    // Define matrix dimensions
    int rows_A, cols_A, rows_B, cols_B;
    printf("Enter dimensions of matrix A (rows columns): ");
    scanf("%d %d", &rows_A, &cols_A);
    printf("Enter dimensions of matrix B (rows columns): ");
    scanf("%d %d", &rows_B, &cols_B);

    // Check if dimensions are valid for matrix multiplication
    if (cols_A != rows_B) {
        printf("Invalid dimensions for matrix multiplication!\n");
        return 0;
    }

    // Allocate memory for matrices A, B, and C
    float *h_A = (float *)malloc(rows_A * cols_A * sizeof(float));
    float *h_B = (float *)malloc(rows_B * cols_B * sizeof(float));
    float *h_C = (float *)malloc(rows_A * cols_B * sizeof(float));

    // Initialize matrices A and B with random values
    for (int i = 0; i < rows_A * cols_A; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < rows_B * cols_B; i++) {
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Perform sequential matrix multiplication
    struct timeval start, end;
    gettimeofday(&start, NULL);
    matrixMultiplicationSequential(h_A, h_B, h_C, rows_A, cols_A, cols_B);
    gettimeofday(&end, NULL);
    double seqTime = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
    printf("Sequential Runtime: %.3f ms\n", seqTime);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
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

