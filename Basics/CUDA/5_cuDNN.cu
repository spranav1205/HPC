#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#define CHECK_CUDA(call) {cudaError_t err = call; if (err != cudaSuccess) {fprintf(stderr, "CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); exit(err); }}
#define CHECK_CUDNN(call) {cudnnStatus_t stat = call; if (stat != CUDNN_STATUS_SUCCESS) {fprintf(stderr, "cuDNN Error: %s (err_num=%d)\n", cudnnGetErrorString(stat), stat); exit(stat); }}

void cpu_convolution(float* input, float* kernels, float* output, int width, int height, int inChannels, int outChannels, int kernelSize, int stride=1, int padding=0)
{
    int outWidth = (width - kernelSize + 2 * padding) / stride + 1;
    int outHeight = (height - kernelSize + 2 * padding) / stride + 1;

    for (int oc = 0; oc < outChannels; oc++)
    {
        for (int oh = 0; oh < outHeight; oh++)
        {
            for (int ow = 0; ow < outWidth; ow++)
            {
                float sum = 0.0f;
                for (int ic = 0; ic < inChannels; ic++)
                {
                    for (int kh = 0; kh < kernelSize; kh++)
                    {
                        for (int kw = 0; kw < kernelSize; kw++)
                        {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                            {
                                sum += input[ic * height * width + ih * width + iw] *
                                       kernels[oc * inChannels * kernelSize * kernelSize + ic * kernelSize * kernelSize + kh * kernelSize + kw];
                            }
                        }
                    }
                }
                output[oc * outHeight * outWidth + oh * outWidth + ow] = sum;
            }
        }
    }
}

__global__ void naiveConv2d(float* input, float* kernels, float* output, int* width, int* height, int inChannels, int outChannels, int kernelSize, int stride=1, int padding=0)
{
    int outWidth = (*width - kernelSize + 2 * padding) / stride + 1;
    int outHeight = (*height - kernelSize + 2 * padding) / stride + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = outWidth * outHeight * outChannels;

    if (idx < totalThreads)
    {
        int oc = idx / (outWidth * outHeight);
        int rem = idx % (outWidth * outHeight);
        int oh = rem / outWidth;
        int ow = rem % outWidth;

        float sum = 0.0f;
        for (int ic = 0; ic < inChannels; ic++)
        {
            for (int kh = 0; kh < kernelSize; kh++)
            {
                for (int kw = 0; kw < kernelSize; kw++)
                {
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;
                    if (ih >= 0 && ih < *height && iw >= 0 && iw < *width)
                    {
                        sum += input[ic * (*height) * (*width) + ih * (*width) + iw] *
                               kernels[oc * inChannels * kernelSize * kernelSize + ic * kernelSize * kernelSize + kh * kernelSize + kw];
                    }
                }
            }
        }
        output[oc * outHeight * outWidth + oh * outWidth + ow] = sum;
    }
}

int generateRandomFloat(float min, float max)
{
    return min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));
}

const int NUM_WARMUP = 3;
const int NUM_RUNS = 10;

int main()
{
    CHECK_CUDA(cudaSetDevice(0));
    const int inChannels = 16;
    const int outChannels = 32;
    const int width = 512;
    const int height = 512;
    const int kernelSize = 5;
    const int padding = 2;
    const int stride = 1;
    const int outWidth = (width - kernelSize + 2 * padding) / stride + 1;
    const int outHeight = (height - kernelSize + 2 * padding) / stride + 1;
    const int inputSize = inChannels * width * height;
    const int kernelSizeTotal = outChannels * inChannels * kernelSize * kernelSize;
    const int outputSize = outChannels * outWidth * outHeight;
    float *h_input = (float*)malloc(inputSize * sizeof(float));
    float *h_kernels = (float*)malloc(kernelSizeTotal * sizeof(float));
    float *h_output_cpu = (float*)malloc(outputSize * sizeof(float));
    float *h_output_naive = (float*)malloc(outputSize * sizeof(float));
    float *h_output_cudnn = (float*)malloc(outputSize * sizeof(float)); 

    for (int i = 0; i < inputSize; i++)
        h_input[i] = generateRandomFloat(0.0f, 1.0f);
    for (int i = 0; i < kernelSizeTotal; i++)
        h_kernels[i] = generateRandomFloat(-1.0f, 1.0f);

    // ============ CPU CONVOLUTION ============
    printf("\n=== CPU CONVOLUTION ===\n");
    
    printf("Running CPU convolution (single run)...\n");
    std::vector<float> cpu_times;
    clock_t start_cpu = clock();
    cpu_convolution(h_input, h_kernels, h_output_cpu, width, height, inChannels, outChannels, kernelSize, stride, padding);
    clock_t end_cpu = clock();
    float cpu_milliseconds = ((float)(end_cpu - start_cpu) / CLOCKS_PER_SEC) * 1000.0f;
    cpu_times.push_back(cpu_milliseconds);

    float cpu_avg = 0.0f, cpu_min = cpu_times[0], cpu_max = cpu_times[0];
    for (float t : cpu_times)
    {
        cpu_avg += t;
        cpu_min = (t < cpu_min) ? t : cpu_min;
        cpu_max = (t > cpu_max) ? t : cpu_max;
    }
    cpu_avg /= NUM_RUNS;

    printf("CPU Convolution - Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms\n", cpu_avg, cpu_min, cpu_max);

    // GPU memory allocation
    float *d_input, *d_kernels, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernels, kernelSizeTotal * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, outputSize * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernels, h_kernels, kernelSizeTotal * sizeof(float), cudaMemcpyHostToDevice));

    // Naive Convolution Setup
    int h_width[1] = {width};
    int h_height[1] = {height};
    int *d_width, *d_height;

    CHECK_CUDA(cudaMalloc(&d_width, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_height, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_width, h_width, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_height, h_height, sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockSize(256);  // Use 256 threads per block for 1D kernel
    int numThreads = outWidth * outHeight * outChannels;
    int gridSizeX = (numThreads + blockSize.x - 1) / blockSize.x;
    dim3 gridSize(gridSizeX, 1);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ============ NAIVE CONVOLUTION ============
    printf("\n=== NAIVE CONVOLUTION ===\n");
    
    // Warmup runs
    printf("Warmup runs...\n");
    for (int i = 0; i < NUM_WARMUP; i++)
    {
        naiveConv2d<<<gridSize, blockSize>>>(d_input, d_kernels, d_output, d_width, d_height, inChannels, outChannels, kernelSize, stride, padding);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark runs
    std::vector<float> naive_times;
    printf("Benchmark runs...\n");
    for (int run = 0; run < NUM_RUNS; run++)
    {
        CHECK_CUDA(cudaEventRecord(start));
        naiveConv2d<<<gridSize, blockSize>>>(d_input, d_kernels, d_output, d_width, d_height, inChannels, outChannels, kernelSize, stride, padding);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        naive_times.push_back(milliseconds);
    }

    CHECK_CUDA(cudaMemcpy(h_output_naive, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    float naive_avg = 0.0f, naive_min = naive_times[0], naive_max = naive_times[0];
    for (float t : naive_times)
    {
        naive_avg += t;
        naive_min = (t < naive_min) ? t : naive_min;
        naive_max = (t > naive_max) ? t : naive_max;
    }
    naive_avg /= NUM_RUNS;

    printf("GPU Naive Convolution - Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms\n", naive_avg, naive_min, naive_max);

    // ============ cuDNN CONVOLUTION ============
    printf("\n=== cuDNN CONVOLUTION ===\n");

    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, inChannels, height, width));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, outChannels, outHeight, outWidth));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outChannels, inChannels, kernelSize, kernelSize));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    // ===== ALGORITHM SELECTION (NOT TIMED) =====
    printf("Finding best algorithm...\n");
    // Find best algorithm
    int returnedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returnedAlgos;
    cudnnConvolutionFwdAlgoPerf_t perfResults[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        inputDesc,
        filterDesc,
        convDesc,
        outputDesc,
        returnedAlgoCount,
        &returnedAlgos,
        perfResults));

    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
    for (int i = 1; i < returnedAlgos; i++)
    {
        if (perfResults[i].status == CUDNN_STATUS_SUCCESS && perfResults[i].time < perfResults[i-1].time)
            algo = perfResults[i].algo;
    }

    printf("Selected Algorithm: %d\n", algo);

    size_t workspaceBytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        inputDesc,
        filterDesc,
        convDesc,
        outputDesc,
        algo,
        &workspaceBytes));

    void* d_workspace = nullptr;
    if (workspaceBytes > 0)
        CHECK_CUDA(cudaMalloc(&d_workspace, workspaceBytes));

    const float alpha = 1.0f, beta = 0.0f;

    // ===== TIMING STARTS HERE =====
    // Warmup runs
    printf("Warmup runs...\n");
    for (int i = 0; i < NUM_WARMUP; i++)
    {
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            inputDesc, d_input,
            filterDesc, d_kernels,
            convDesc,
            algo,
            d_workspace, workspaceBytes,
            &beta,
            outputDesc, d_output));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark runs
    std::vector<float> cudnn_times;
    printf("Benchmark runs...\n");
    for (int run = 0; run < NUM_RUNS; run++)
    {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            inputDesc, d_input,
            filterDesc, d_kernels,
            convDesc,
            algo,
            d_workspace, workspaceBytes,
            &beta,
            outputDesc, d_output));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        cudnn_times.push_back(milliseconds);
    }

    CHECK_CUDA(cudaMemcpy(h_output_cudnn, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    float cudnn_avg = 0.0f, cudnn_min = cudnn_times[0], cudnn_max = cudnn_times[0];
    for (float t : cudnn_times)
    {
        cudnn_avg += t;
        cudnn_min = (t < cudnn_min) ? t : cudnn_min;
        cudnn_max = (t > cudnn_max) ? t : cudnn_max;
    }
    cudnn_avg /= NUM_RUNS;

    printf("GPU cuDNN Convolution - Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms\n", cudnn_avg, cudnn_min, cudnn_max);

    // ============ COMPARISON ============
    printf("\n=== RESULTS ===\n");
    printf("Configuration: Input %dx%dx%d, Kernels %dx%d, Output %dx%d\n", width, height, inChannels, kernelSize, kernelSize, outWidth, outHeight);
    printf("Padding: %d, Stride: %d\n", padding, stride);
    printf("\nTiming Summary:\n");
    printf("  CPU Naive    - Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms\n", cpu_avg, cpu_min, cpu_max);
    printf("  GPU Naive    - Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms\n", naive_avg, naive_min, naive_max);
    printf("  GPU cuDNN    - Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms\n", cudnn_avg, cudnn_min, cudnn_max);
    printf("\nSpeedup vs CPU:\n");
    printf("  GPU Naive: %.2fx\n", cpu_avg / naive_avg);
    printf("  GPU cuDNN: %.2fx\n", cpu_avg / cudnn_avg);
    printf("\nSpeedup GPU Naive vs cuDNN: %.2fx\n", naive_avg / cudnn_avg);
    
    // ============ VERIFICATION ============
    bool cpu_vs_naive_match = true;
    bool cpu_vs_cudnn_match = true;
    bool naive_vs_cudnn_match = true;
    float max_diff_cpu_naive = 0.0f;
    float max_diff_cpu_cudnn = 0.0f;
    float max_diff_naive_cudnn = 0.0f;
    const float tolerance = 1e-2f;
    
    for (int i = 0; i < outputSize; i++)
    {
        float diff_cn = fabs(h_output_naive[i] - h_output_cudnn[i]);
        float diff_cpu_n = fabs(h_output_cpu[i] - h_output_naive[i]);
        float diff_cpu_c = fabs(h_output_cpu[i] - h_output_cudnn[i]);
        
        max_diff_naive_cudnn = (diff_cn > max_diff_naive_cudnn) ? diff_cn : max_diff_naive_cudnn;
        max_diff_cpu_naive = (diff_cpu_n > max_diff_cpu_naive) ? diff_cpu_n : max_diff_cpu_naive;
        max_diff_cpu_cudnn = (diff_cpu_c > max_diff_cpu_cudnn) ? diff_cpu_c : max_diff_cpu_cudnn;
        
        if (diff_cn > tolerance)
            naive_vs_cudnn_match = false;
        if (diff_cpu_n > tolerance)
            cpu_vs_naive_match = false;
        if (diff_cpu_c > tolerance)
            cpu_vs_cudnn_match = false;
    }
    
    printf("\n=== OUTPUT VERIFICATION (Against CPU Baseline) ===\n");
    if (cpu_vs_naive_match)
        printf("✓ CPU vs GPU Naive match! (max diff: %.2e)\n", max_diff_cpu_naive);
    else
        printf("✗ CPU vs GPU Naive differ! (max diff: %.2e, tolerance: %.2e)\n", max_diff_cpu_naive, tolerance);
    
    if (cpu_vs_cudnn_match)
        printf("✓ CPU vs GPU cuDNN match! (max diff: %.2e)\n", max_diff_cpu_cudnn);
    else
        printf("✗ CPU vs GPU cuDNN differ! (max diff: %.2e, tolerance: %.2e)\n", max_diff_cpu_cudnn, tolerance);
    
    if (naive_vs_cudnn_match)
        printf("✓ GPU Naive vs GPU cuDNN match! (max diff: %.2e)\n", max_diff_naive_cudnn);
    else
        printf("✗ GPU Naive vs GPU cuDNN differ! (max diff: %.2e, tolerance: %.2e)\n", max_diff_naive_cudnn, tolerance);

    // ============ CLEANUP ============
    printf("\nCleaning up...\n");
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernels));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_width));
    CHECK_CUDA(cudaFree(d_height));
    if (d_workspace)
        CHECK_CUDA(cudaFree(d_workspace));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(outputDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    free(h_input);
    free(h_kernels);
    free(h_output_cpu);
    free(h_output_naive);
    free(h_output_cudnn);

    printf("Done!\n");
    return 0;
}