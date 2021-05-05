
#include <stdlib.h>
#include <stdio.h>

#include "common.h"

void MPI_double_layer_convolution(int M, int N, float **input, int K1, float **kernel1, int K2, float **kernel2, float **output) {

    double temp;
    float **output_kernel1;

    alloc2D(&output_kernel1, M - K1 + 1, N - K1 + 1);

    // 1. Calculate input 2d array with kernel 1
    for (int i = 0; i <= M-K1; i++) {
        for (int j = 0; j <= N-K1; j++) {
            temp = 0.0;
            for (int k=0; k < K1; k++) {
                for (int l = 0; l < K1; l++) {
                    temp += input[i + k][j + l] * kernel1[k][l];
                }
            }
            output_kernel1[i][j] = temp;
        }
    }

    printf("OUTPUT_KERNEL1 2D ARRAY:\n");
    printmat(output_kernel1, M - K1 + 1, N - K1 + 1);

    // 2. Calculate result of step 1 with kernel 2
    for (int i = 0; i <= (M - K1 + 1) - K2; i++) {
        for (int j = 0; j <= (N - K1 + 1) - K2; j++) {
            temp = 0.0;
            for (int k=0; k < K2; k++) {
                for (int l = 0; l < K2; l++) {
                    temp += output_kernel1[i + k][j + l] * kernel2[k][l];
                }
            }
            output[i][j] = temp;
        }
    }

    printf("OUTPUT 2D ARRAY:\n");
    printmat(output, M - K1 - K2 + 2, N - K1 - K2 + 2);

}
