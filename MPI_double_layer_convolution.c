
#include <stdlib.h>
#include <stdio.h>

#include "common.h"

void mat_mul(int M, int N, int K, float **input, float **kernel, float **output) {

    double temp;

    for (int i = 0; i <= M-K; i++) {
        for (int j = 0; j <= N-K; j++) {
            temp = 0.0;
            for (int k=0; k < K; k++) {
                for (int l = 0; l < K; l++) {
                    temp += input[i + k][j + l] * kernel[k][l];
                }
            }
            output[i][j] = temp;
        }
    }

}

void MPI_double_layer_convolution(int M, int N, float **input, int K1, float **kernel1, int K2, float **kernel2, float **output, int *sendcounts, int *scatter_displacements, int *n_rows, int *gather_displacements, int my_rank) {


    // Scatter input to do first calculation with kernel 1
    MPI_Scatterv(input, sendcounts, scatter_displacements, MPI_FLOAT, input, N*n_rows[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    float **output_kernel1;
    allocate_2d_array(&output_kernel1, M - K1 + 1, N - K1 + 1);
    mat_mul(n_rows[my_rank], N, K1, input, kernel1, output_kernel1);
    MPI_Gatherv(output_kernel1, n_rows[my_rank], MPI_FLOAT, output_kernel1, n_rows, gather_displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Scatter output of first kernel calculation to do the second calculation with kernel 2
    MPI_Scatterv(output_kernel1, sendcounts, scatter_displacements, MPI_FLOAT, output_kernel1, N*n_rows[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    mat_mul((M - K1 + 1), (N - K1 + 1), K2, output_kernel1, kernel2, output);
    MPI_Gatherv(output, n_rows[my_rank], MPI_FLOAT, output, n_rows, gather_displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);

    free_2d_array(output_kernel1);

}
