# Matrix-Multiplication-using-MPI
C implementation of Matrix Multiplication in MPI. This work was made for the Concordia University's COMP-6231 Distributed System Design Course.

# Getting Started
To run this code you need to install MPI [Open MPI](https://www.open-mpi.org/).

## Compilation
    $ mpicc -o exec ./MatrixMultiplicationUsingMPI.c 
    
## Execution
To execute four multiple parallel processes.

    $ mpirun -np 4 ./exec
  
