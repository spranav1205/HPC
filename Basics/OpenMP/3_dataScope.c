#include <stdio.h>
#include <omp.h>

int global_counter = 100;

#pragma omp threadprivate(global_counter) // Each thread has its own copy of 'global_counter'


//TODO: add nice examples :/

int main() {
    int factor = 5;

    printf("Master global_counter before: %d\n", global_counter);

    // --- FIRST PARALLEL REGION ---
    // copyin: Copies the master's 'global_counter' (100) to all threads.
    // firstprivate: Copies 'factor' (5) to all threads' local scratchpads.
    // firstprivate vs copyin: firstprivate is for regular variables, copyin is specifically for threadprivate variables.
    #pragma omp parallel copyin(global_counter) firstprivate(factor) num_threads(2)
    {
        int tid = omp_get_thread_num();
        global_counter += (tid + 1); // Thread 0: 101, Thread 1: 102
        factor += 10;                // Local factor becomes 15
        
        printf("Thread %d: global_counter=%d, factor=%d\n", tid, global_counter, factor);
    }

    printf("\nMaster factor after (unchanged): %d\n", factor);

    // --- SECOND PARALLEL REGION ---
    // Note: No 'copyin' here! 
    // threadprivate variables REMEMBER their values from the previous region!
    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        printf("Thread %d still sees global_counter as: %d\n", tid, global_counter);
    }

    return 0;
}