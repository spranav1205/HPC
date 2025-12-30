#include <mpi.h>

void parallel_vector_sum(double local_x[], double local_y[], double local_z[], int local_n)
{
    for(int local_i = 0; local_i < local_n; local_i++ )
    {
        local_z[local_i] = local_x[local_i] + local_y[local_i];
    }
}

void Read_Vector(int my_rank, int local_n, int n, char vec_name[], double local_a[], MPI_Comm comm)
    {
    // MPI_Scatter(send_buffer, send_count, send_type, receive_buffer, receive_count, receive_type, root, communicator)
    // Rank 0 divides array 'a' into chunks of 'local_n' elements and distributes them
    // Each process (including root) receives 'local_n' elements into 'local_a' in rank order
    // send_count is per-process, so rank 0 sends local_n elements to rank 0, local_n to rank 1, etc.

    double* a = NULL;
    int i;
    
    if (my_rank == 0) {
        a = (double*)malloc(n*sizeof(double));
        printf("Enter the vector %s\n", vec_name);
        for(i = 0; i < n; i++)
        {
            scanf("%lf", &a[i]);
        }
        MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, comm); 
        free(a);
    }
    else
    {
        MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, comm);
    }
}

void Print_Vector (int my_rank, int local_n, int n, char vec_name[], double local_b[], double b[], MPI_Comm comm)
{
    // MPI_Gather(send_buffer, send_count, send_type, receive_buffer, receive_count, receive_type, root, communicator)
    // Each process sends 'local_n' elements from 'local_b' to rank 0
    // Rank 0 collects 'local_n' elements from each process into consecutive positions in 'b'
    // Total elements gathered at root = comm_sz * local_n

    if(my_rank == 0)
    {
        MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE, 0, comm);
        printf("%s\n", vec_name);

        for (int i=0; i<n; i++)
        {
            printf("%f ", b[i]);
        }
    }
    else
    {
        MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE, 0, comm);
    }
}

void sum_local(double local_a[], double local_b[], double local_sum[], int local_n)
{
    for (int i=0; i<local_n; i++)
    {
        local_sum[i] = local_a[i] + local_b[i];
    }
}

int main()
{
    int comm_sz; // Total number of processes
    int my_rank;

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int local_n = 10/comm_sz; // Assuming divisible
    double a[local_n];
    double b[local_n];

    char vec_x[2] = {'x', '\0'};
    char vec_y[2] = {'y', '\0'}; 
    char vec_s[4] = {'s','u','m','\0'};

    Read_Vector(my_rank, local_n, 10, vec_x, a, MPI_COMM_WORLD);
    Read_Vector(my_rank, local_n, 10, vec_y, b, MPI_COMM_WORLD);

    double local_sum[local_n];
    double sum[10];
    sum_local(a,b,local_sum, local_n);

    //Gather
    Print_Vector(my_rank, local_n, 10, vec_s, local_sum, sum, MPI_COMM_WORLD);

    MPI_Finalize();
}