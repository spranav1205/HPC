#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

void CHECK_CUDA(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s (err_num=%d)\n",
                cudaGetErrorString(err), err);
        exit(err);
    }
}

void CHECK_CUBLAS(cublasStatus_t stat)
{
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUBLAS Error: %d\n", stat);
        exit(stat);
    }
}

void cpuMatMul(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * N + j] = 0;
            for (int k = 0; k < K; k++)
            {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void printMatrix(const float* A, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", A[i * cols + j]);
        }
        printf("\n");
    }
}

int main()
{
    const int M = 4;
    const int N = 4;
    const int K = 4;

    float h_A[M * K] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    float h_B[K * N] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    float h_C_cpu[M * N] = {0};
    float h_C_gpu_fp32[M * N] = {0};
    float h_C_gpu_fp16[M * N] = {0};

    printf("Matrix A:\n");
    printMatrix(h_A, M, K);
    printf("Matrix B:\n");
    printMatrix(h_B, K, N);

    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    CHECK_CUDA(cudaMalloc(&d_A_fp32, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_fp32, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_fp32, M * N * sizeof(float)));

    __half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    CHECK_CUDA(cudaMalloc(&d_A_fp16, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B_fp16, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C_fp16, M * N * sizeof(__half)));

    CHECK_CUDA(cudaMemcpy(d_A_fp32, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp32, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<half> h_A_fp16(M * K);
    std::vector<half> h_B_fp16(K * N);

    for (int i = 0; i < M * K; i++)
        h_A_fp16[i] = __float2half(h_A[i]); // Convert float to half
    for (int i = 0; i < K * N; i++)
        h_B_fp16[i] = __float2half(h_B[i]);

    CHECK_CUDA(cudaMemcpy(d_A_fp16, h_A_fp16.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp16, h_B_fp16.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));

    // Create CUBLAS handle (needed for all CUBLAS operations)
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // Matrix Layouts
    // For marices given by A[M][K], B[K][N], C[M][N]
    // cublas expects column-major order thus we swap M and K

    // Matrix Layout for FP32
    cublasLtMatrixLayout_t layoutA_fp32, layoutB_fp32, layoutC_fp32;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA_fp32, CUDA_R_32F, K, M, K));  
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB_fp32, CUDA_R_32F, N, K, N));  
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC_fp32, CUDA_R_32F, N, M, N));  

    // Matrix Layout for FP16
    cublasLtMatrixLayout_t layoutA_fp16, layoutB_fp16, layoutC_fp16;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA_fp16, CUDA_R_16F, K, M, K));  
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB_fp16, CUDA_R_16F, N, K, N));  
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC_fp16, CUDA_R_16F, N, M, N));

    // Create Matmul Descriptors (This defines computation type)
    cublasLtMatmulDesc_t matmulDesc_fp32, matmulDesc_fp16;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp32, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp16, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    
    // Set matrix operations for A and B
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp32, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp32, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Matrix multiplication: C = alpha * A * B + beta * C

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // FP32 Matmul
    CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc_fp32, &alpha, d_A_fp32, layoutA_fp32, d_B_fp32, layoutB_fp32, &beta, nullptr, layoutC_fp32, d_C_fp32, layoutC_fp32, nullptr, nullptr, 0, nullptr));
    
    const half alpha_h = __float2half(1.0f);
    const half beta_h = __float2half(0.0f);

    // FP16 Matmul
    CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc_fp16, &alpha_h, d_A_fp16, layoutA_fp16, d_B_fp16, layoutB_fp16, &beta_h, nullptr, layoutC_fp16, d_C_fp16, layoutC_fp16, nullptr, nullptr, 0, nullptr));
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_C_gpu_fp32, d_C_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<half> h_C_fp16(M * N);
    CHECK_CUDA(cudaMemcpy(h_C_fp16.data(), d_C_fp16, M * N * sizeof(__half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; i++)
        h_C_gpu_fp16[i] = __half2float(h_C_fp16[i]); // Convert half to float   
    
    // CPU Matrix Multiplication for verification
    cpuMatMul(h_A, h_B, h_C_cpu, M, N, K);

    // Compare CPU and GPU results
    bool match_fp32 = true;
    bool match_fp16 = true;
    const float epsilon = 1e-3f;

    for (int i = 0; i < M * N; i++)
    {
        if (fabs(h_C_cpu[i] - h_C_gpu_fp32[i]) > epsilon)
        {
            match_fp32 = false;
            break;
        }
        if (fabs(h_C_cpu[i] - h_C_gpu_fp16[i]) > epsilon)
        {
            match_fp16 = false;
            break;
        }
    }

    printf("GPU FP32 Result:\n");
    printMatrix(h_C_gpu_fp32, M, N);
    printf("GPU FP16 Result:\n");
    printMatrix(h_C_gpu_fp16, M, N);

    // Cleanup
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutA_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutB_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutC_fp32));    
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutA_fp16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutB_fp16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutC_fp16));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc_fp32));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc_fp16));
    CHECK_CUBLAS(cublasLtDestroy(handle));

    CHECK_CUDA(cudaFree(d_A_fp32));
    CHECK_CUDA(cudaFree(d_B_fp32)); 
    CHECK_CUDA(cudaFree(d_C_fp32));
    CHECK_CUDA(cudaFree(d_A_fp16));
    CHECK_CUDA(cudaFree(d_B_fp16));
    CHECK_CUDA(cudaFree(d_C_fp16));
    
    if (match_fp32)
        printf("FP32 GPU result matches CPU result.\n");
    else
        printf("FP32 GPU result does NOT match CPU result.\n");
    
    if (match_fp16)
        printf("FP16 GPU result matches CPU result.\n");
    else
        printf("FP16 GPU result does NOT match CPU result.\n");

    /*
    The matrix layouts (K,M,K), (N,K,N), (N,M,N) tell cuBLAS to interpret your row-major data as if it were column-major by swapping dimensions. 
    This effectively transposes how the data is read—so your MxK row-major matrix is seen as KxM column-major, and similarly for B and C.
    transA = CUBLAS_OP_N means "don't transpose" (no-op flag). 
    With the swapped dimensions from the layout, cuBLAS performs the correct matrix multiplication: 
    it reads the data in the transposed orientation the layout defined, 
    achieving the right computation without explicitly calling a transpose function.
    */
}