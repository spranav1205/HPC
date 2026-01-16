#include <stdio.h>
#include <cuda.h>


__global__ void index_kernel()
{
    int block_id = 
    blockIdx.x +        // Apartment number on this floor
    blockIdx.y * gridDim.x +        // Floor number times apartments per floor
    blockIdx.z * gridDim.x * gridDim.y;      // Building number times apartments per building
    
    int thread_id = 
    threadIdx.x + 
    threadIdx.y * blockDim.x + 
    threadIdx.z * blockDim.x * blockDim.y;

    int block_offset = block_id * blockDim.x * blockDim.y * blockDim.z; // Total threads per block times block ID

    int global_id = thread_id + block_offset; // Global thread ID

    printf("block: (%d,%d,%d) = %d thread: (%d,%d,%d),  global_id: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, block_id, threadIdx.x, threadIdx.y, threadIdx.z, global_id);


}

int main()
{
    //Grid
    const int b_x = 4;
    const int b_y = 4;
    const int b_z = 4;

    //Block
    const int t_x = 2;
    const int t_y = 2;
    const int t_z = 2;
    
    int blocks = b_x * b_y * b_z;
    int threads = t_x * t_y * t_z;

    dim3 gridDim(b_x, b_y, b_z);
    dim3 blockDim(t_x, t_y, t_z);

    printf("Launching kernel with %d blocks of %d threads (%d total threads)\n", blocks, threads, blocks * threads);

    index_kernel<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}