#include <iostream>
#include <stdio.h>
#include <omp.h> 
#include <vector>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

int main()
{
    int size  = 24;
    int n = 8;

    std::vector<int> array(size);

    #pragma omp parallel num_threads(n) // No. of threads limited to 16 (dependends on device)
    {
        int id = omp_get_thread_num();
        int block = CEIL_DIV(size,n);

        int iter;
        if (id == n-1)
        {
            iter = size % (block);

            if (iter == 0)
            {
                iter = block;
            }
        }
        else 
        {
            iter = block;
        }

        #pragma omp critical
        {
            std::cout << "Hello from thread " << id << "\n";
        }

        #pragma unroll(4) // Unroll depends on instruction look ahead of the architecture (nothing to do with threads)
        for (int i=0; i<iter; i++)
        {
            array[id*block + i] = (int) 10 * (float)rand()/float(RAND_MAX);
        }
    }

    for(int j = 0; j < size; j++)
    {
        std::cout<<array[j]<<" ";
    }

    

}