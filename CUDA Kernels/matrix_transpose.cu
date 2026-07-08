#include <iostream>
#include <cuda_runtime.h>

void iniitialize_matrix(int rows, int cols, float *matrix)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix[i * cols + j] = static_cast<float>(i * cols + j);
        }
    }
}

__global__ void matrix_transpose_naive(int rows, int cols, float *input, float *output)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols)
    {
        output[j * rows + i] = input[i * cols + j];
    }
}

__global__ void matrix_transpose_tiled(int rows, int cols, float *input, float *output)
{
    __shared__ float tile[16][16 + 1]; 

    int x_in = blockIdx.x * blockDim.x + threadIdx.x;
    int y_in = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_in < cols && y_in < rows)
    {
        tile[threadIdx.y][threadIdx.x] = input[y_in * cols + x_in];
    }

    __syncthreads();

    int x_out = blockIdx.y * blockDim.y + threadIdx.x; 
    int y_out = blockIdx.x * blockDim.x + threadIdx.y;

    if (x_out < rows && y_out < cols)
    {
        output[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void matrix_transpose(int rows, int cols, float *input, float *output)
{
    __shared__ float tile[16][16 + 1]; 

    int x_in = (blockIdx.x * 16) + (threadIdx.x * 4);
    int y_in = (blockIdx.y * 16) + threadIdx.y;   
    int smem_x = threadIdx.x * 4; 

    if (x_in < cols && y_in < rows)
    {
        
        float4 *input4 = (float4 *)&input[y_in * cols + x_in];
        float4 reg = input4[0];

        tile[threadIdx.y][smem_x] = reg.x;
        tile[threadIdx.y][smem_x + 1] = reg.y;
        tile[threadIdx.y][smem_x + 2] = reg.z;
        tile[threadIdx.y][smem_x + 3] = reg.w;
    }

    __syncthreads();

    int x_out = (blockIdx.y * 16) + (threadIdx.x * 4);
    int y_out = (blockIdx.x * 16) + threadIdx.y;

    if (x_out < rows && y_out < cols)
    {
        float4 reg_out = make_float4(
            tile[smem_x][threadIdx.y],
            tile[smem_x + 1][threadIdx.y],
            tile[smem_x + 2][threadIdx.y],
            tile[smem_x + 3][threadIdx.y]
        );

        float4 *output4 = (float4 *)&output[y_out * rows + x_out];
        *output4 = reg_out;
    }
}


int main()
{
    float *d_input, *d_output;

    int rows = 1024;
    int cols = 2048;

    size_t size = rows * cols * sizeof(float);

    d_input = (float *)malloc(size);
    d_output = (float *)malloc(size);

    iniitialize_matrix(rows, cols, d_input);

    float *gpu_input, *gpu_output;

    cudaMalloc((void **)&gpu_input, size);
    cudaMalloc((void **)&gpu_output, size);

    cudaMemcpy(gpu_input, d_input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(4, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrix_transpose<<<gridSize, blockSize>>>(rows, cols, gpu_input, gpu_output);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken by GPU: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(d_output, gpu_output, size, cudaMemcpyDeviceToHost);

    // Verify the result
    bool correct = true;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (d_output[j * rows + i] != d_input[i * cols + j])
            {
                correct = false;
                break;
            }
        }
    }

    if (correct)
    {
        std::cout << "Matrix transpose is correct." << std::endl;
    }
    else
    {
        std::cout << "Matrix transpose is incorrect." << std::endl;
    }

    free(d_input);
    free(d_output);
}
