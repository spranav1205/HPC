#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int shared_total = 0;
    int work_array[8] = {0};

    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();

        work_array[tid] = tid * 5; 
        work_array[tid + 4] = (tid + 4) * 5;

        // --- BARRIER ---
        #pragma omp barrier

        // 2. ATOMIC PHASE (High Performance)
        #pragma omp atomic
        shared_total += (work_array[tid] + work_array[tid + 4]);

        // --- BARRIER ---
        // Ensure the total is fully calculated before anyone prints the final result.
        #pragma omp barrier

        // 3. CRITICAL PHASE (Mutual Exclusion)
        // Only one thread prints at a time to prevent garbled console output.
        #pragma omp critical
        {
            printf("Thread %d reporting: Current Global Total is %d\n", tid, shared_total);
        }
    }
    return 0;
}