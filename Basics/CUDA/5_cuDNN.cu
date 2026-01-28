#include <cuda_runtime.h>
// #include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) {cudaError_t err = call; if (err != cudaSuccess) {fprintf(stderr, "CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); exit(err); }}
#define CHECK_CUDNN(call) {cudnnStatus_t stat = call; if (stat != CUDNN_STATUS_SUCCESS) {fprintf(stderr, "cuDNN Error: %s (err_num=%d)\n", cudnnGetErrorString(stat), stat); exit(stat); }}

__global__ void naiveConv2d(float* input, float* kernels, float* output, int* width, int* height, int inChannels, int outChannels, int kernelSize)
{
    int threadsPerBlock = blockDim.x * blockDim.y;
    int idx = blockIdx.y * gridDim.x * threadsPerBlock + blockIdx.x * threadsPerBlock + threadIdx.y * blockDim.x + threadIdx.x;
    int outWidth = width[0] - kernelSize + 1;
    int outHeight = height[0] - kernelSize + 1;

    if(idx < outWidth * outHeight * outChannels)
    {
        // Calculate output pixel coordinates thus each thread computes one output pixel
        int ch = idx/(outHeight*outWidth);
        int i = (idx%(outHeight*outWidth))/outWidth;
        int j = (idx%outWidth);

        float sum = 0.0f;
        for (int c = 0; c < inChannels; c++)
        {
            for (int ki = 0; ki < kernelSize; ki++)
            {
                for (int kj = 0; kj < kernelSize; kj++)
                {
                    sum += input[c * height[0] * width[0] + (i + ki) * width[0] + (j + kj)] *
                           kernels[ch * inChannels * kernelSize * kernelSize + c * kernelSize * kernelSize + ki * kernelSize + kj];
                }
            }
        }
        output[ch * outHeight * outWidth + i * outWidth + j] = sum;
    }
}

int generateRandomFloat(float min, float max)
{
    return min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));
}

int main()
{
    CHECK_CUDA(cudaSetDevice(0));
    const int inChannels = 3;
    const int outChannels = 16;
    const int width = 32;
    const int height = 32;
    const int kernelSize = 3;
    const int outWidth = width - kernelSize + 1;
    const int outHeight = height - kernelSize + 1;
    const int inputSize = inChannels * width * height;
    const int kernelSizeTotal = outChannels * inChannels * kernelSize * kernelSize;
    const int outputSize = outChannels * outWidth * outHeight;
    float *h_input = (float*)malloc(inputSize * sizeof(float));
    float *h_kernels = (float*)malloc(kernelSizeTotal * sizeof(float));
    float *h_output_naive = (float*)malloc(outputSize * sizeof(float));
    float *h_output_cudnn = (float*)malloc(outputSize * sizeof(float)); 

    for (int i = 0; i < inputSize; i++)
        h_input[i] = generateRandomFloat(0.0f, 1.0f);
    for (int i = 0; i < kernelSizeTotal; i++)
        h_kernels[i] = generateRandomFloat(-1.0f, 1.0f);

    float *d_input, *d_kernels, *d_output;

    CHECK_CUDA(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernels, kernelSizeTotal * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, outputSize * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernels, h_kernels, kernelSizeTotal * sizeof(float), cudaMemcpyHostToDevice));

    int h_width[1] = {width};
    int h_height[1] = {height}; 
    int *d_width, *d_height;

    CHECK_CUDA(cudaMalloc(&d_width, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_height, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_width, h_width, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_height, h_height, sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    int numThreads = outWidth * outHeight * outChannels;
    dim3 gridSize((numThreads + blockSize.x * blockSize.y - 1) / (blockSize.x * blockSize.y), 1);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    naiveConv2d<<<gridSize, blockSize>>>(d_input, d_kernels, d_output, d_width, d_height, inChannels, outChannels, kernelSize);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaMemcpy(h_output_naive, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Naive Convolution Time: %f ms\n", milliseconds);

}
