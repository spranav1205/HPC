#include <mpi.h>

void matrix_vect_mult(double local_A[], double x[], double local_y[], int local_m, int n)
{
    int local_i, j;
    int local_ok = 1;

    for (local_i = 0; local_i < local_m; local_i++)
    {
        for (j = 0; j < n; j++)
        {
            local_y[local_i] += local_A[local_i * n + j] * x[j];
        }
    }
}

int main()
{
    int comm_sz, my_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Fixed matrix dimensions
    int m = 8, n = 8;  // 8x8 matrix
    int local_m = m / comm_sz;  // rows per process

    // Fixed matrix A (row-major): identity matrix for simplicity
    double* A = NULL;
    if (my_rank == 0) {
        A = (double*)malloc(m * n * sizeof(double));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = (i == j) ? 2.0 : 0.0;  // Identity matrix
    }

    // BROADCAST X
    double* x = (double*)malloc(n * sizeof(double));
    if (my_rank == 0) {
        for (int i = 0; i < n; i++)
            x[i] = i + 1.0;
    }

    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // SCATTER A, Y
    double* local_A = (double*)malloc(local_m * n * sizeof(double));
    MPI_Scatter(A, local_m * n, MPI_DOUBLE, local_A, local_m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* local_y = (double*)malloc(local_m * sizeof(double));
    for (int i = 0; i < local_m; i++)
        local_y[i] = 0.0;

    // Perform matrix-vector multiplication
    matrix_vect_mult(local_A, x, local_y, local_m, n);

    // GATHER
    double* y = NULL;
    if (my_rank == 0)
        y = (double*)malloc(m * sizeof(double));
    MPI_Gather(local_y, local_m, MPI_DOUBLE, y, local_m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print result on rank 0
    if (my_rank == 0) {
        printf("Result vector y:\n");
        for (int i = 0; i < m; i++)
            printf("%lf\n", y[i]);
        free(y);
        free(A);
    }

    free(x);
    free(local_A);
    free(local_y);

    MPI_Finalize();
    return 0;
}