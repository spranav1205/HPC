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


int main(void)
{
    double partial_integral; // NOT a reference?!
    int comm_sz; // Total number of processes
    int my_rank;

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double a = 0;
    double b = M_PI;
    double steps = 100000;

    double step_size = (b-a)/steps;

    int per_process_steps = (int) steps/(comm_sz-1); 

    if(my_rank != 0)
    {
        double start = a + (b-a)*(my_rank - 1)/(comm_sz-1);
        double stop = a + (b-a)*(my_rank)/(comm_sz-1);
        
        partial_integral = trapezoidal(fn,start, stop, per_process_steps);
        printf("Process %d finished its job!, partial sum is %f\n", my_rank, partial_integral);

        MPI_Send(&partial_integral, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); //Sends data to the main process
    }
    else {
        printf("Greetings from master process %d of %d!, I will now compute the integral:", my_rank, comm_sz);
        printf("Steps in each integral: %d", per_process_steps);
        double integral = 0.0;
        for (int q = 1; q < comm_sz; q++) 
        {
            MPI_Recv(&partial_integral, 1, MPI_DOUBLE, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            integral += partial_integral;
        }

        printf("The value of the integral is: %f", integral);
    }

    MPI_Finalize();
    return 0;
    
}