#include <stdio.h>
#include <cuda.h>

__global__
void hello_kernel()
{
  printf("Hello from GPU! Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main(void)
{
  printf("Hello from CPU!\n");
  
  // Launch kernel with 2 blocks, 5 threads per block
  hello_kernel<<<2, 5>>>();
  
  // Wait for GPU to finish
  cudaDeviceSynchronize();
  
  printf("Done!\n");
  
  return 0;
}
