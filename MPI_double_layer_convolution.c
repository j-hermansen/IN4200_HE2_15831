
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
                    printf("input[%d][%d]\n", i + k, j + l);
                    temp += input[i + k][j + l] * kernel[k][l];
                }
            }
            output[i][j] = temp;
        }
    }

}

void MPI_double_layer_convolution(int M, int N, float **input, int K1, float **kernel1, int K2, float **kernel2, float **output, int *sendcounts, int *Sdispls, int *n_rows, int *Gdispls, int my_rank) {

    double temp;

    // Scatter input to do first calculation with kernel 1
    MPI_Scatterv(input, sendcounts, Sdispls, MPI_FLOAT, input, N*n_rows[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    float **output_kernel1;
    allocate_2d_array(&output_kernel1, M - K1 + 1, N - K1 + 1);
    mat_mul(n_rows[my_rank], N, K1, input, kernel1, output_kernel1);
    MPI_Gatherv(output_kernel1, n_rows[my_rank], MPI_FLOAT, output_kernel1, n_rows, Gdispls, MPI_FLOAT, 0, MPI_COMM_WORLD);

    printf("OUTPUT_KERNEL1 2D ARRAY:\n");
    print_2d_array(output_kernel1, M - K1 + 1, N - K1 + 1);

    MPI_Scatterv(output_kernel1, sendcounts, Sdispls, MPI_FLOAT, output_kernel1, N*n_rows[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    mat_mul(((M - K1 + 1) - K2), ((N - K1 + 1) - K2), K2, output_kernel1, kernel2, output);
    MPI_Gatherv(output_kernel1, n_rows[my_rank], MPI_FLOAT, output_kernel1, n_rows, Gdispls, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("OUTPUT 2D ARRAY:\n");
        print_2d_array(output, M - K1 - K2 + 2, N - K1 - K2 + 2);
    }

    free_2d_array(output_kernel1);

}
