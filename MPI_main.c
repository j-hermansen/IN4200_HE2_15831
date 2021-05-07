
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include "common.h"
#include "MPI_double_layer_convolution.c"

int main(int nargs, char **args) {

    int M=0, N=0, K1=0, K2=0, my_rank, number_of_processes;
    float **input=NULL, **output=NULL, **kernel1=NULL, **kernel2=NULL;

    // Initialize MPI, and get process id and total processes
    MPI_Init(&nargs, &args);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);


    if (my_rank == 0) {

        if (nargs != 5) {
            printf("Wrong number of arguments. Program needs 4 arguments, Usage: ./program M N K1 K2\n");
            return EXIT_FAILURE;
        } else {
            M = atoi(args[1]);    // M rows in input matrix
            N = atoi(args[2]);    // N columns in input matrix
            K1 = atoi(args[3]);   // K1 rows and columns in first kernel matrix
            K2 = atoi(args[4]);   // K2 rows and columns in second kernel matrix
        }

        allocate_2d_array(&input, N, M);                           // Allocate 2D array input with M rows and N columns
        allocate_2d_array(&output, N-K1-K2+2, M-K1-K2+2);    // Allocate 2D array output with M-K1-K2+2 rows and N-K1-K2+2 columns
        allocate_2d_array(&kernel1, K1, K1);                       // Allocate first kernel with K1 rows and K1 columns
        allocate_2d_array(&kernel2, K2, K2);                       // Allocate second kernel with K2 rows and K2 columns

        // Fill input 2D array with random values
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                input[i][j] = uniform() * 3;
            }
        }

        // Fill kernel1 with random values
        for (size_t i = 0; i < K1; i++) {
            for (size_t j = 0; j < K1; j++) {
                kernel1[i][j] = uniform() * 2;
            }
        }

        // Fill kernel2 with random values
        for (size_t i = 0; i < K2; i++) {
            for (size_t j = 0; j < K2; j++) {
                kernel2[i][j] = uniform() * 2;
            }
        }

//        /// Print out 2D arrays
//        printf("INPUT 2D ARRAY:\n");
//        printmat(input, N, M);
//        printf("\n");
//        printf("KERNEL1 2D ARRAY:\n");
//        printmat(kernel1, K1, K1);
//        printf("\n");
//        printf("KERNEL2 2D ARRAY:\n");
//        printmat(kernel2, K2, K2);
//        printf("\n");

    }

//    MPI_Barrier(MPI_COMM_WORLD);

    int *n_rows = malloc(number_of_processes * sizeof *n_rows);           // Should hold number of rows per process (a row is here the row size of K1 or K2)
    int *sendcounts = malloc(number_of_processes * sizeof *sendcounts);   // Should hold count of types that's being scattered
    int *scatter_displacements = malloc(number_of_processes * sizeof *scatter_displacements);         // Should
    int *gather_displacements = malloc(number_of_processes * sizeof *gather_displacements);         // Should
    int rows = (N / number_of_processes);                                      // Number of rows per process (some will get additional rows). It should be chunks of K1 rows.
    int remainder = N % number_of_processes;                                   // Should hold the remainder when N is no t divisable with number of processes

    scatter_displacements[0] = 0;
    gather_displacements[0] = 0;

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);      /// Process 0 should broadcast value M to all other processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);      /// Process 0 should broadcast value N to all other processes
    MPI_Bcast(&K1, 1, MPI_INT, 0, MPI_COMM_WORLD);     /// Process 0 should broadcast value K1 to all other processes
    MPI_Bcast(&K2, 1, MPI_INT, 0, MPI_COMM_WORLD);     /// Process 0 should broadcast value K2 to all other processes

    MPI_Bcast(&n_rows, number_of_processes, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank > 0) {
        allocate_2d_array(&kernel1, K1, K1);                     /// Allocated kernel 1 with K1 rows and K1 columns
        allocate_2d_array(&kernel2, K2, K2);                     /// Allocated kernel 2 with K2 rows and K2 columns
    }

    for (int rank = 0; rank < number_of_processes-1; rank++) {
        n_rows[rank] = rows + ((rank >= (number_of_processes - remainder)) ? 1:0);   /// Set number of rows for a given process
        sendcounts[rank] = n_rows[rank]*N;                                 /// Set count of data types to send for given process
        scatter_displacements[rank+1] = scatter_displacements[rank] + (sendcounts[rank] - 2*N);
        gather_displacements[rank+1] = gather_displacements[rank] + n_rows[rank];
    }
    n_rows[number_of_processes-1] = rows + ((number_of_processes-1) >= (number_of_processes - remainder) ? 1:0);   /// Set number of rows for last process
    sendcounts[number_of_processes-1] = n_rows[number_of_processes-1]*N;                                 /// Set count of data types to send for last process


    MPI_Bcast(&kernel1, K1*K1, MPI_FLOAT, 0, MPI_COMM_WORLD);            /// Process 0 should broadcast kernel 1 to all other processes
    MPI_Bcast(&kernel2, K2*K2, MPI_FLOAT, 0, MPI_COMM_WORLD);            /// Process 0 should broadcast kernel 2 to all other processes


    if (my_rank == 0) {
        printf("PRINT OUT n_rows VALUES:\n");
        for (int i = 0; i < number_of_processes; ++i) {
            printf("n_rows[%d]: %d\n", i, n_rows[i]);
        }
    }


    double start = MPI_Wtime();;

    MPI_double_layer_convolution(M, N, input, K1, kernel1, K2, kernel2, output, sendcounts, scatter_displacements, n_rows, gather_displacements, my_rank);

    double end = MPI_Wtime();

    printf("Time used: %lf\n", end - start);



    // Should compare parallel with sequential execution
    if (my_rank==0) {

        float **output_sequential1;
        allocate_2d_array(&output_sequential1, N-K1+1, M-K1+1);
        single_layer_convolution(M, N, input, K1, kernel1, output_sequential1);

        float **output_sequential2;
        allocate_2d_array(&output_sequential2, N-K1-K2+2, M-K1-K2+2);
        single_layer_convolution(M, N, output_sequential1, K2, kernel2, output_sequential2);

        printf("SEQUENTIAL 2D ARRAY OUTPUT:\n");
        print_2d_array(output_sequential2, N-K1-K2+2, M-K1-K2+2);

        free_2d_array(output_sequential1);
        free_2d_array(output_sequential2);

    }

    free_2d_array(input);
    free_2d_array(output);
    free_2d_array(kernel1);
    free_2d_array(kernel2);

    free(n_rows);
    free(scatter_displacements);
    free(gather_displacements);
    free(sendcounts);


    MPI_Finalize();

    return 0;

}


