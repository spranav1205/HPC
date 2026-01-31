#include <stdio.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <string.h>

int main(int argc, char** argv)
{
    // Check for explanation flag
    int showExplanations = 0;
    if (argc > 1 && (strcmp(argv[1], "--explain") == 0 || strcmp(argv[1], "-e") == 0)) {
        showExplanations = 1;
    }
    
    // Get the number of CUDA-capable devices
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        printf("No CUDA devices found.\n");
        return 1;
    }

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    // Iterate through each device and print its properties
    for (int i = 0; i < deviceCount; i++)
    {
        // Set the current device
        cudaSetDevice(i);

        // Structure to hold device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // Device name
        printf("Device %d: %s\n", i, prop.name);
        
        // Core properties
        if (showExplanations) printf("\n=== CORE PROPERTIES ===\n");
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        if (showExplanations) printf("    > Version of the CUDA architecture. Higher = more features/performance.\n");
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        if (showExplanations) printf("    > Maximum number of threads that can exist in a single block.\n");
        printf("  Threads per Warp: %d\n", prop.warpSize);
        if (showExplanations) printf("    > Number of threads that execute together in lockstep (SIMT execution unit).\n");
        printf("  Warp Allocation Granularity: 4\n");
        if (showExplanations) printf("    > Warps are allocated in groups of this size to a block.\n");
        
        // Register information
        if (showExplanations) printf("\n=== REGISTER PROPERTIES ===\n");
        printf("  Max Regs per Block: %d\n", prop.regsPerBlock);
        if (showExplanations) printf("    > Total 32-bit registers available for a single thread block.\n");
        printf("  Max Regs per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
        if (showExplanations) printf("    > Total registers available across all blocks on a streaming multiprocessor (SM).\n");
        printf("  Reg Allocation Unit Size: 256\n");
        if (showExplanations) printf("    > Registers are allocated in chunks of this size per warp.\n");
        printf("  Reg Allocation Granularity: warp\n");
        if (showExplanations) printf("    > Register allocation happens at the warp level, not per thread.\n");
        
        // Memory information
        if (showExplanations) printf("\n=== MEMORY PROPERTIES ===\n");
        printf("  Total Global Mem: %d MB\n", (int)(prop.totalGlobalMem / (1024 * 1024)));
        if (showExplanations) printf("    > Total device memory (VRAM) accessible by all threads.\n");
        printf("  Max Shared Mem per Block: %d KB\n", (int)(prop.sharedMemPerBlock / 1024));
        if (showExplanations) printf("    > Fast on-chip memory shared by threads in a block. Used for inter-thread communication.\n");
        printf("  CUDA Runtime Shared Mem Overhead per Block: 1024 B\n");
        if (showExplanations) printf("    > Reserved shared memory used by CUDA runtime for each block.\n");
        printf("  Shared Mem per Multiprocessor: %d B\n", (int)prop.sharedMemPerMultiprocessor);
        if (showExplanations) printf("    > Total shared memory available on each SM, divided among resident blocks.\n");
        
        // Multiprocessor information
        if (showExplanations) printf("\n=== MULTIPROCESSOR PROPERTIES ===\n");
        printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);
        if (showExplanations) printf("    > Number of streaming multiprocessors (SMs). Each SM can run multiple blocks concurrently.\n");
        
        // Calculate max warps and threads per multiprocessor
        // For Compute Capability >= 2.0, typically supports 48 or 64 warps per SM
        int maxWarpsPerMultiprocessor = 48; // Default for most modern architectures
        if (prop.major >= 7) {
            maxWarpsPerMultiprocessor = (prop.major == 7 && prop.minor < 5) ? 64 : 
                                        (prop.major == 8 && prop.minor >= 6) ? 48 : 64;
        }
        int maxThreadsPerMultiprocessor = maxWarpsPerMultiprocessor * prop.warpSize;
        printf("  Max Threads per Multiprocessor: %d\n", maxThreadsPerMultiprocessor);
        if (showExplanations) printf("    > Maximum threads that can be resident on a single SM at once.\n");
        printf("  Max Warps per Multiprocessor: %d\n", maxWarpsPerMultiprocessor);
        if (showExplanations) printf("    > Maximum warps that can be scheduled on a single SM simultaneously.\n");
        
        // Additional details
        if (showExplanations) printf("\n=== ADDITIONAL PROPERTIES ===\n");
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        if (showExplanations) printf("    > Width of the memory interface. Wider = higher bandwidth.\n");
        printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
        if (showExplanations) printf("    > Size of the L2 cache shared by all SMs for global memory accesses.\n");
        printf("  Max Blocks per Multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
        if (showExplanations) printf("    > Maximum number of thread blocks that can reside on a single SM.\n");
        printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        if (showExplanations) printf("    > Whether the device can execute multiple kernels simultaneously.\n");
        printf("  Max Grid Dimensions: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        if (showExplanations) printf("    > Maximum size of the grid in (x, y, z) dimensions.\n");
        printf("  Max Block Dimensions: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        if (showExplanations) printf("    > Maximum size of a thread block in (x, y, z) dimensions.\n");
        printf("\n");
    }

    return 0;
}