#include <stdio.h>
#include <string.h>
#include <mpi.h>

const int MAX_STRING = 100;

int main(void)
{
    char greeting[MAX_STRING];
    int comm_sz; // Total number of processes
    int my_rank;

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if(my_rank != 0)
    {
        sprintf(greeting, "Greetings from process %d of %d!", my_rank, comm_sz); //Writes to the string greeting 
        MPI_Send(greeting, strlen(greeting)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD); //Sends data to the main process

        // Message buffer, message size , data type, destination, tag, communicator
    }
    else {
        printf("Greetings from master process %d of %d!\n, I will now recieve messages from other processes:", my_rank, comm_sz);
        for (int q = 1; q < comm_sz; q++) 
        {
            MPI_Recv(greeting, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // message buffer, size, datatype, source, tag, communicator, status

            printf("%s\n", greeting);
        }
    }

    MPI_Finalize();
    return 0;
}
