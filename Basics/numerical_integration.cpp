#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <functional>

double fn(double x)
{
    return exp(x);
}

template <typename F>
double trapezoidal(F&& f, double a, double b, int steps)
{
    const double h = (b - a) / steps;

    double sum = 0.5 * (f(a) + f(b));

    for (int i = 1; i < steps; ++i)
    {
        sum += f(a + i * h);
    }

    return h * sum;
}

void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p, int* n_p)
    {
    if (my_rank == 0) {
    printf("Enter a, b, and n\n");
    scanf("%lf %lf %d", a_p, b_p, n_p);
    }
    MPI_Bcast(a_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

int main(void)
{
    double partial_integral; // NOT a reference?!
    int comm_sz; // Total number of processes
    int my_rank;

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double a, b;
    int steps;
    Get_input(my_rank, comm_sz,&a, &b, &steps);

    double step_size = (b-a)/steps;

    int per_process_steps = steps/(comm_sz); 
    int remainder = steps % comm_sz;

    double start = a + (b-a)*(my_rank)/(comm_sz);
    double stop = a + (b-a)*(my_rank+1)/(comm_sz);

    if(my_rank ==  comm_sz-1)
    {
        stop = b;
        per_process_steps += remainder;
    }
           
    partial_integral = trapezoidal(fn,start, stop, per_process_steps);
    
    printf("Process %d finished its job!, partial sum is %f\n", my_rank, partial_integral);

    double total_integral = 0.0;

    MPI_Reduce(&partial_integral, &total_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // Pre built: input, output, count?, datatype, method, communication channel
    
    if (my_rank == 0)
    {
        printf("\nFinal integral result: %.10f\n", total_integral);
    }

    MPI_Finalize();
    return 0;
    
}