
#ifndef IN4200_HE2_15831_COMMON_H
#define IN4200_HE2_15831_COMMON_H

#define uniform() (rand() / (RAND_MAX + 1.0))

#include <stdlib.h>
#include <stdio.h>

int alloc2D(float ***A, int n, int m) {
    *A = malloc(n * sizeof *A);
    (*A)[0] = malloc(n*m * sizeof (*A)[0]);
    if (!(*A)[0] || !*A){
        // Allocation failed.
        printf("Allocation failed\n");
        return 1;
    }

    for (size_t i = 1; i < n; i++) {
        (*A)[i] = &((*A)[0][i*m]);
    }
    return 0;
}


// Free pointers allocated in alloc2D.
int free2D(float **A) {
    free(A[0]);
    free(A);
    return 0;
}

// Print matrix values.
int printmat(float **A, int n, int m)
{
    for (size_t i = 0; i < n; ++i){
        printf("|");
        for (size_t j = 0; j < m; ++j){
            printf("%4.0lf ", A[i][j]);
        }
        printf("|\n");
    }
}

void single_layer_convolution (int M, int N, float **input, int K, float **kernel, float **output) {
    int i,j,ii,jj;
    double temp;
    for (i=0; i<=M-K; i++)
        for (j=0; j<=N-K; j++) {
            temp = 0.0;
            for (ii=0; ii<K; ii++)
                for (jj=0; jj<K; jj++)
                    temp += input[i+ii][j+jj]*kernel[ii][jj];
            output[i][j] = temp;
        }
}

#endif
