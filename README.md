# MPI parallelization of convolution computation

### IN3200/IN4200 Home Exam 2, Spring 2021



Made using C programming language and MPI. Comments about
Include three files:
- **MPI_main.c**
  - *Main program where execution starts*
- **common.h**
  - *Include common function used by multiple files*
- **MPI_double_layer_convolution.c**
  - *A function for calculating double layer convolution with MPI as a tool to better performance * 
    

This program is made in WSL (Windows Subsystem for Linux) using VIM text editor. 

Compile with:
```bash
mpicc MPI_main.c
```

Run with:
```bash
mpirun -np <number of processes> ./a.out M N K1 K2
```
where 
- M is the number of rows in input 2d array
- N is the number of columns in input 2d array
- K1 is the size (row and column) of the first kernel
- K2 is the size (row and column) of the second kernel





