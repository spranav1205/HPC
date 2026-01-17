#include <stdio.h>
#include <cuda.h>

void print_peak_memory() {
    size_t free_mem, total_mem, used_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    used_mem = total_mem - free_mem;
    printf("Peak GPU Memory Used: %.2f MB / %.2f MB\n", 
           used_mem / (1024.0 * 1024.0), 
           total_mem / (1024.0 * 1024.0));
}

void cpu_matmul(float *A, float *B, float *C, int m, int k, int n)
{
    // Dimensions of A are (m,k), B are (k,n) and thus C are (m,n)

    for (int a=0; a<m; a++)
    {
        for (int b=0; b<n; b++)
        {
            for (int i=0; i<k; i++)
            {
                *(C + (a*n) + b) += (*(A + (a*k) + i)) * (*(B + (i*n) + b));
            }
        }
    }
}

__global__ void gpu_matmul (float *A, float *B, float *C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        for (int i=0; i<k; i++)
        {
            *(C + (row*n) + col) += (*(A + (row*k) + i)) * (*(B + (i*n) + col));
        }
    }
}

void initialize_matrix(float *mat, int rows, int cols, float value)
{
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            *(mat + i*cols + j) = value;
        }
    }
}

int main()
{
    float *h_A, *h_B, *h_C;
    int m = 4000, k = 4000, n = 4000;

    h_A = (float*)malloc(m * k * sizeof(float));
    h_B = (float*)malloc(k * n * sizeof(float));
    h_C = (float*)malloc(m * n * sizeof(float));

    initialize_matrix(h_A, m, k, 1.0f);
    initialize_matrix(h_B, k, n, 2.0f);

    // Time

    clock_t start, end;
    double cpu_time_used, gpu_time_used;

    start = clock();

    cpu_matmul(h_A, h_B, h_C, m, k, n);

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU Time taken: %f seconds\n", cpu_time_used);

    // printf("Result matrix C:\n");
    // for (int i=0; i<m; i++)
    // {
    //     for (int j=0; j<n; j++)
    //     {
    //         printf("%f ", *(h_C + i*n + j));
    //     }
    //     printf("\n");
    // }

    // GPU Part

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, m * n * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y); // Ceiling division

    start = clock();

    gpu_matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);
    cudaDeviceSynchronize();
    end = clock();
    gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("GPU Time taken: %f seconds\n", gpu_time_used);
    
    print_peak_memory();

    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("Result matrix C from GPU:\n");
    // for (int i=0; i<m; i++)
    // {
    //     for (int j=0; j<n; j++)
    //     {
    //         printf("%f ", *(h_C + i*n + j));   
    //     }
    //     printf("\n");
    // }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
    
}