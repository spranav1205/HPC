#include <iostream>
#include <omp.h> // Required header for OpenMP

int main() {
    // 1. Check environment
    int max_threads = omp_get_max_threads();
    std::cout << "System can spawn up to " << max_threads << " threads.\n";

    // 2. The FORK: Creating the parallel region
    // 'parallel' starts the team of threads
    // 'num_threads(4)' explicitly sets the count (optional)
    #pragma omp parallel num_threads(4)
    {
        // Everything inside these braces is executed by EVERY thread
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        #pragma omp critical
        {
            // 'critical' ensures only one thread prints at a time
            std::cout << "Hello from thread " << thread_id 
                      << " out of " << total_threads << "\n";
        }
        
        // 3. The JOIN: Implicit barrier here
        // Threads wait for each other before leaving the curly brace
    }

    std::cout << "Back to the master thread. Parallel region finished.\n";
    return 0;
}