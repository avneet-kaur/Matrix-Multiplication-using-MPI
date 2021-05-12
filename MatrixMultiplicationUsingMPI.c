#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include <time.h>
#include <stdbool.h>

#define NRA 4         /* number of rows in matrix A */
#define NCA 3         /*number of columns in matrix A*/
#define NCB 7         /*number of columns in matrix B*/
#define MASTER 0      /* process ID of first process */
#define FROM_MASTER 1 /* setting a message TAG */
#define FROM_WORKER 2 /* setting a message TAG */

double A[NRA][NCA], B[NCA][NCB], C[NRA][NCB], res[NRA][NCB];

void generate_matrixA(double A[NRA][NCA])
{
    //The rand() function generates numbers from 0 to RAND_MAX, the value of which is system dependent.
    srand(time(NULL));
    for (int i = 0; i < NRA; i++)
    {
        for (int j = 0; j < NCA; j++)
        {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }
}

void generate_matrixB(double B[NCA][NCB])
{
    srand(time(NULL));
    for (int i = 0; i < NCA; i++)
    {
        for (int j = 0; j < NCB; j++)
        {
            B[i][j] = rand() % 10;
        }
    }
}

void print_matrixA(double result[NRA][NCA])
{
    for (int i = 0; i < NRA; i++)
    {
        for (int j = 0; j < NCA; j++)
        {
            printf("%.0f\t", result[i][j]);
        }
        printf("\n");
    }
}

void print_matrixB(double result[NCA][NCB])
{
    for (int i = 0; i < NCA; i++)
    {
        for (int j = 0; j < NCB; j++)
        {
            printf("%.0f\t", result[i][j]);
        }
        printf("\n");
    }
}

void print_matrixC(double result[NRA][NCB])
{
    for (int i = 0; i < NRA; i++)
    {
        for (int j = 0; j < NCB; j++)
        {
            printf("%.0f\t", result[i][j]);
        }
        printf("\n");
    }
}

void sequentialMultiplication(double A[NRA][NCA], double B[NCA][NCB], double res[NRA][NCB])
{
    clock_t begin = clock();
    int i, j, k;
    for (i = 0; i < NRA; i++)
    {
        for (j = 0; j < NCB; j++)
        {
            res[i][j] = 0;
            for (k = 0; k < NCA; k++)
                res[i][j] += A[i][k] * B[k][j];
        }
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("\n");
    printf("\n/************ SEQUENTIAL RESULT /************\n\n");
    print_matrixC(res);

    printf("\nExecution Time of Sequential Matrix Multiplication Code = %lu seconds \n", (end - begin));
}

int main(int argc, char **argv)
{
    MPI_Status status;
    int errorCode,
        extraRows,
        messageTag,
        totalNoOfProcesses,
        processId, totalWorkerTasks, source, dest, rows, offset;

    /*----------------*/
    /* initialize MPI */
    /*----------------*/

    MPI_Init(&argc, &argv);

    /* Each process gets unique ID (rank)*/
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    /* Number of processes in communicator -> totalNoOfProcesses*/
    MPI_Comm_size(MPI_COMM_WORLD, &totalNoOfProcesses);

    if (totalNoOfProcesses < 2)
    {
        printf("\nNeed at least two MPI processes. Killing all processes.....\n");
        /*invoking MPI_ABORT causes Open MPI to kill all MPI processes.*/
        MPI_Abort(MPI_COMM_WORLD, errorCode);
        exit(1);
    }

    totalWorkerTasks = totalNoOfProcesses - 1;

    /*-------------------------------------------------------*/
    /********************** Master Process *******************/
    /*-------------------------------------------------------*/

    if (processId == MASTER)
    {
        printf("\nProgram started with %d worker processes.\n", totalWorkerTasks);
        double startTime = MPI_Wtime();

        /* Matrix A and Matrix B both will be generated with random numbers*/
        generate_matrixA(A);
        generate_matrixB(B);

        printf("\n\t\tMatrix Multiplication using MPI\n");
        printf("\nMatrix A\n\n");
        print_matrixA(A);
        printf("\nMatrix B\n\n");
        print_matrixB(B);
        sequentialMultiplication(A, B, res);

        /*--------------------------------------*/
        /* send matrix data to the worker tasks */
        /*--------------------------------------*/

        /* determining fraction of array i.e:- rows of Matrix A to be processed by workers */
        rows = NRA / totalWorkerTasks;

        /*left out array*/
        extraRows = NRA % totalWorkerTasks;

        /*Offset is the starting point of the row of Matrix A which is to be sent to the worker*/
        offset = 0;

        messageTag = FROM_MASTER;

        /* To each worker send : Start point, number of rows to process, and sub-arrays to process */
        for (dest = 1; dest <= totalWorkerTasks; dest++)
        {
            //printf("\nNumber of extra rows :- %d\n", extraRows);

            rows = (dest <= extraRows) ? rows + 1 : rows;

            // printf("\nSending %d rows to process ID: %d with offset=%d\n",
            //        rows, dest, offset);

            MPI_Send(&offset, 1, MPI_INT, dest, messageTag, MPI_COMM_WORLD);

            //number of rows to be sent
            MPI_Send(&rows, 1, MPI_INT, dest, messageTag, MPI_COMM_WORLD);

            //send corresponding parts of matrix A to each worker
            MPI_Send(&A[offset][0], rows * NCA, MPI_DOUBLE, dest, messageTag,
                     MPI_COMM_WORLD);

            /*sending matrix B*/
            MPI_Send(&B, NCA * NCB, MPI_DOUBLE, dest, messageTag, MPI_COMM_WORLD);

            /*Offset is modified according to number of rows sent to each process*/
            offset = offset + rows;
            extraRows = 0;
        }

        messageTag = FROM_WORKER;

        /* wait for results from all worker tasks */
        /* Receive results from worker tasks and will be blocked until all the workers send their calculated result */
        for (int i = 1; i <= totalWorkerTasks; i++)
        {
            source = i;
            /*Receive the starting point of particular worker process */
            MPI_Recv(&offset, 1, MPI_INT, source, messageTag, MPI_COMM_WORLD, &status);

            /*Receive the number of rows that each worker process processed */
            MPI_Recv(&rows, 1, MPI_INT, source, messageTag, MPI_COMM_WORLD, &status);

            /* C matrix containing calculates rows of each worker process */
            MPI_Recv(&C[offset][0], rows * NCB, MPI_DOUBLE, source, messageTag, MPI_COMM_WORLD, &status);

            printf("\nReceived results from process ID: %d\n", source);
        }

        /*Print the result matrix*/
        printf("\n/************ PARALLEL RESULT /************\n\n");
        print_matrixC(C);
        printf("\n");

        double endTime = MPI_Wtime();
        printf("Execution Time of Parallel Matrix Multiplication Code = %f seconds on %d nodes. \n", (endTime - startTime), totalNoOfProcesses);
    }

    /*-------------------------------------------------------*/
    /********************** Worker Processes ****************/
    /*-----------------------------------------------------*/

    if (processId > MASTER)
    {
        messageTag = FROM_MASTER;

        /*worker process receive the initial offset position of matrix A */
        MPI_Recv(&offset, 1, MPI_INT, MASTER, messageTag, MPI_COMM_WORLD,
                 &status);

        /*receives number of rows sent by root process  */
        MPI_Recv(&rows, 1, MPI_INT, MASTER, messageTag, MPI_COMM_WORLD,
                 &status);

        /* receive the matrix A starting at offset */
        MPI_Recv(&A, rows * NCA, MPI_DOUBLE, MASTER, messageTag, MPI_COMM_WORLD,
                 &status);

        /* receive the matrix B */
        MPI_Recv(&B, NCA * NCB, MPI_DOUBLE, MASTER, messageTag, MPI_COMM_WORLD,
                 &status);

        /* Calculate the product and store result in C */
        for (int k = 0; k < NCB; k++)
        {
            for (int i = 0; i < rows; i++)
            {
                C[i][k] = 0.0;
                for (int j = 0; j < NCA; j++)
                    C[i][k] = C[i][k] + A[i][j] * B[j][k];
            }
        }

        messageTag = FROM_WORKER;

        /* send the starting point of calculated matrix C */
        MPI_Send(&offset, 1, MPI_INT, MASTER, messageTag, MPI_COMM_WORLD);

        /* send the number of rows of C */
        MPI_Send(&rows, 1, MPI_INT, MASTER, messageTag, MPI_COMM_WORLD);

        /* send the final Matrix C */
        MPI_Send(&C, rows * NCB, MPI_DOUBLE, MASTER, messageTag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
