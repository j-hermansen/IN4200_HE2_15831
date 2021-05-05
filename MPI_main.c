
#include <stdlib.h>
#include <stdio.h>

#include "common.h"
#include "MPI_double_layer_convolution.c"


int main() {

    // TODO: SHOULD TAKE M, N, K1, K2 FROM PROGRAM ARGUMENTS
    int M=6, N=6, K1=3, K2=3, my_rank;
    float **input=NULL, **output=NULL, **kernel1=NULL, **kernel2=NULL;

    // TODO: ONLY FOR TESTING - SHOULD BE ALLOCATED IN FUNCTION @ PROCESS 0
    alloc2D(&input, N, M);
    alloc2D(&output, N-K1-K2+2, M-K1-K2+2);
    alloc2D(&kernel1, K1, K1);
    alloc2D(&kernel2, K2, K2);

    // Initialize input with some values.
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            input[i][j] = (int)(uniform() * 3);
        }
    }

    // Initialize kernels with some values.
    for (size_t i = 0; i < K1; i++) {
        for (size_t j = 0; j < K1; j++) {
            kernel1[i][j] = (int)(uniform() * 2);
            kernel2[i][j] = (int)(uniform() * 2);
        }
    }

    printf("INPUT 2D ARRAY:\n");
    printmat(input, N, M);
    printf("\n");
    printf("KERNEL1 2D ARRAY:\n");
    printmat(kernel1, K1, K1);
    printf("\n");
    printf("KERNEL2 2D ARRAY:\n");
    printmat(kernel2, K2, K2);
    printf("\n");



//    MPI_Init(&nargs, &args);
//    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank==0) {
        // read from command line the values of M, N, and K
        // allocate 2D array ’input’ with M rows and N columns
        // allocate 2D array ’output’ with M-K+1 rows and N-K+1 columns
        // allocate the convolutional kernel with K rows and K columns
        // fill 2D array ’input’ with some values
        // fill kernel with some values
        // ....
    }

    // process 0 broadcasts values of M, N, K1, K2 to all the other processes
    // ...
    if (my_rank>0) {
        // allocated the convolutional kernels with K1 or K2 rows and K1 or K2 columns
        // ...
    }

    // process 0 broadcasts the content of kernels to all the other processes
    // ...
    // parallel computation of a double-layer convolution
    MPI_double_layer_convolution (M, N, input, K1, kernel1, K2, kernel2, output);

    if (my_rank==0) {
        // For example, compare the content of array ’output’ with that is
        // produced by the two sequential function single_layer_convolution calls
        // ...
    }

    free2D(input);
    free2D(output);
    free2D(kernel1);
    free2D(kernel2);


//    MPI_Finalize();

    return 0;

}


