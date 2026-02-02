#include <stdio.h>
#include <cuda.h>
#include <assert.h>

#define BLOCKSIZE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void naive_matmul(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) 
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

__global__ void coalescing_matmul(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) 
{
    // Each warp computes a BLOCKSIZE x BLOCKSIZE sub-matrix
    // threadIdx.x/BLOCKSIZE gives row index within the sub-matrix
    // threadIdx.x%BLOCKSIZE gives column index within the sub-matrix
    // Thus, for threads in the same warp, (threadIdx.x/BLOCKSIZE) is same, leading to coalesced accesses in A

    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

__global__ void cacheblocking_matmul(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    const int threadRow = threadIdx.x / BLOCKSIZE;  // 0..31
    const int threadCol = threadIdx.x % BLOCKSIZE;  // 0..31

    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;

    float tmp = 0.0f;

    __shared__ float Asub[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bsub[BLOCKSIZE * BLOCKSIZE];

    // Move base pointers to block tile
    A += blockRow * BLOCKSIZE * K;
    B += blockCol * BLOCKSIZE;
    C += blockRow * BLOCKSIZE * N + blockCol * BLOCKSIZE;

    for (int bk = 0; bk < K; bk += BLOCKSIZE) {

        Asub[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bsub[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        #pragma unroll // Unroll the loop for better performance
        for (int j = 0; j < BLOCKSIZE; ++j)
            tmp += Asub[threadRow * BLOCKSIZE + j] * Bsub[j * BLOCKSIZE + threadCol];

        __syncthreads();
    }

    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

template<const int BM, const int BN, const int BK, const int TM>
__global__ void oneD_blockTiling_matmul(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) 
{
    // Global thread index within the block tile
    const int threadRow = threadIdx.x / BN;  
    const int threadCol = threadIdx.x % BN;  

    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;

    __shared__ float Asub[BM * BK];
    __shared__ float Bsub[BK * BN];

    // Move base pointers to block tile
    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    assert((BM / TM) * BN == blockDim.x);

    // Local thread indices within the shared memory tiles
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;    

    float threadResults[TM] = {0.0f};

    for (uint bk = 0; bk < K; bk += BK) {

        // Each thread loads one element of A and one element of B into shared memory
        Asub[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bsub[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint j = 0; j < BK; ++j) //Sideways within BK tile
        {
            float tempB = Bsub[j * BN + threadCol]; 
            for (uint t = 0; t < TM; ++t) // Downward within TM tile
            {
                threadResults[t] += 
                        Asub[(threadRow * TM + t) * BK + j] * tempB; 
        }
        }
        __syncthreads();
    }

    for (int t = 0; t < TM; ++t) {
        C[(innerRowA * TM + t) * N + threadCol] = 
                alpha * threadResults[t] +
                beta * C[(innerRowA * TM + t) * N + threadCol];
    }
}

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void twoD_blockTiling_matmul(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) 
{
    const int threadRow = threadIdx.x / (BN/ TN);  
    const int threadCol = threadIdx.x % (BN/ TN);  

    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;

    __shared__ float Asub[BM * BK];
    __shared__ float Bsub[BK * BN];

    // Move base pointers to block tile
    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    assert(blockDim.x == (BM/TM) * (BN/TN));

    // Thread indices within (BM x BK) and (BK x BN) shared memory tiles
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN; 

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    // register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    const uint strideA = blockDim.x / BK; // Rows within BM*BK tile = 32
    const uint strideB = blockDim.x / BN; // Rows within BK*BN tile = 4

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        
        // Loads elements along BM
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            // Loads: A[0][innerColA], A[32][innerColA]; A[1][innerColA], A[33][innerColA] ...
            Asub[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA]; 
        }
        
        // Loads elements along BN
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            // Loads: B[0][innerColB], B[4][innerColB]; B[1][innerColB], B[5][innerColB] ...
            Bsub[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB]; 
        }
    
        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {

            // load relevant As & Bs entries into registers
            for (uint i = 0; i < TM; ++i) {
                regM[i] = Asub[(threadRow * TM + i) * BK + dotIdx]; 
            }

            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bsub[dotIdx * BN + threadCol * TN + i];
            }

            // perform outer product on register cache, accumulate
            // into threadResults
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }

        __syncthreads();

    }

    // Write back results from threadResults to global memory C
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }
}

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void vectorize_matmul(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) 
{
    // Same update as memory coalescing version
    const int threadRow = threadIdx.x % (BN/ TN);  
    const int threadCol = threadIdx.x / (BN/ TN);  

    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;

    __shared__ float Asub[BM * BK];
    __shared__ float Bsub[BK * BN];

    // Move base pointers to block tile
    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    assert(blockDim.x == (BM/TM) * (BN/TN));

    // Transposed and vectorized (4 at a time) loading of A and B
    const uint innerColA = threadIdx.x / (BK/4);
    const uint innerRowA = threadIdx.x % (BK/4);
    const uint innerColB = threadIdx.x / (BN/4);
    const uint innerRowB = threadIdx.x % (BN/4); 

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    // register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        
        float4 tmp = reinterpret_cast<float4*>(&A[innerRowA * K + innerColA * 4])[0]; // Starting address in 2D tile, 4 elements are imidiately copied
        Asub[innerRowA * BM + innerColA * 4 + 0] = tmp.x;
        Asub[innerRowA * BM + innerColA * 4 + 1] = tmp.y;
        Asub[innerRowA * BM + innerColA * 4 + 2] = tmp.z;
        Asub[innerRowA * BM + innerColA * 4 + 3] = tmp.w;

        reinterpret_cast<float4*>(&Bsub[innerRowB * BN + innerColB * 4])[0] = 
            reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0]; // Faster than pragma unroll + normal loads
    
        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {

            // load relevant As & Bs entries into registers
            for (uint i = 0; i < TM; ++i) {
                regM[i] = Asub[dotIdx * BM + threadCol * TM + i];
            }

            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bsub[dotIdx * BN + threadCol * TN + i];
            }

            // perform outer product on register cache, accumulate
            // into threadResults
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }

        __syncthreads();

    }

    // Write back results from threadResults to global memory C
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            // Create temparary float4 to enable vectorized writeback
            float4 tmp = reinterpret_cast<float4*>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];

            // Update tmp with computed results
            tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
            tmp.y = alpha * threadResults[resIdxM * TN + resIdxN+1] + beta * tmp.y;
            tmp.z = alpha * threadResults[resIdxM * TN + resIdxN+2] + beta * tmp.z;
            tmp.w = alpha * threadResults[resIdxM * TN + resIdxN+3] + beta * tmp.w;

            // Write tmp back to C?
            reinterpret_cast<float4*>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;
        }
    }
}

void print_matrix(const float *mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

void generate_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

void compare_matrices(const float *mat1, const float *mat2, int rows, int cols) {
    const float epsilon = 1e-3;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float diff = fabs(mat1[i * cols + j] - mat2[i * cols + j]);
            if (diff > epsilon) {
                printf("Matrices differ at (%d, %d): %f vs %f\n", i, j, mat1[i * cols + j], mat2[i * cols + j]);
                return;
            }
        }
    }
    printf(" Sort.\n");
}

int main()
{   
    const int M = 4096; // Rows of A and C
    const int N = 4096; // Columns of B and C
    const int K = 4096; // Columns of A and Rows of B
    double flops = 2.0 * M * N * K;  // GFLOPS: 2*M*N*K

    const float alpha = 1.0f;
    const float beta = 0.0f;

    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C_naive = (float*)malloc(M * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    generate_matrix(h_A, M, K);
    generate_matrix(h_B, K, N);
    generate_matrix(h_C, M, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    // CHANGE: initialize C on device from valid host buffer
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // ==== Naive matrix multiplication kernel launch ==== //

    dim3 blockDim(32, 32, 1); // 2D block
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1); // 2D grid
    float time = 0.0f;

    cudaEvent_t startEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);
    cudaEventRecord(startEvent, 0);

    naive_matmul<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);

    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);
    cudaEventElapsedTime(&time, startEvent, endEvent);
    printf("Naive Matrix Multiplication Time: %f ms | GFLOPS: %.1f\n", time, (flops/1e9)/(time));

    cudaMemcpy(h_C_naive, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    //print_matrix(h_C, M, N);

    // ==== Coalescing matrix multiplication kernel launch ==== //

    time = 0.0f;
    dim3 blockDim2(32 * 32); // 1D block with 1024 threads

    cudaEventRecord(startEvent, 0);
    coalescing_matmul<<<gridDim, blockDim2>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);
    cudaEventElapsedTime(&time, startEvent, endEvent);
    printf("Coalescing Matrix Multiplication Time: %f ms | GFLOPS: %.1f |", time, (flops/1e9)/(time));
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    compare_matrices(h_C_naive, h_C, M, N);

    // ==== Cache-blocking matrix multiplication kernel launch ==== //
    time = 0.0f;
    cudaEventRecord(startEvent, 0);
    cacheblocking_matmul<<<gridDim, blockDim2>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);
    cudaEventElapsedTime(&time, startEvent, endEvent);
    printf("Cache-blocking Matrix Multiplication Time: %f ms | GFLOPS: %.1f |", time, (flops/1e9)/(time));
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    compare_matrices(h_C_naive, h_C, M, N);
    //print_matrix(h_C, M, N);

    // ==== 1D block tiling matrix multiplication kernel launch ==== //
    time = 0.0f;

    // CHANGE: reduce tile size so blockDim3 stays <= 1024 threads
    const int BM = 64; // Tile size in M dimension
    const int BN = 64; // Tile size in N dimension
    const int BK = 8; // Tile size in K dimension
    const int TM = 4;  // Threads per tile in M dimension

    dim3 gridDim3(CEIL_DIV(M, BM), CEIL_DIV(N, BN), 1); // 2D grid
    dim3 blockDim3(BM*BN/TM); // Size of tile divided by number of values computed per thread

    cudaEventRecord(startEvent, 0);
    oneD_blockTiling_matmul<BM,BN,BK,TM><<<gridDim3, blockDim3>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);
    cudaEventElapsedTime(&time, startEvent, endEvent);
    printf("1D Block Tiling Matrix Multiplication Time: %f ms | GFLOPS: %.1f |", time, (flops/1e9)/(time));
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    compare_matrices(h_C_naive, h_C, M, N);
    //print_matrix(h_C, M, N);

    // ==== 2D block tiling matrix multiplication kernel launch ==== //
    time = 0.0f;

    const int TN = 4;  // Threads per tile in N dimension

    dim3 gridDim4(CEIL_DIV(M, BM), CEIL_DIV(N, BN), 1); // 2D grid
    dim3 blockDim4((BM*BN)/(TM*TN)); // Size of tile divided by number of values computed per thread

    cudaEventRecord(startEvent, 0);
    twoD_blockTiling_matmul<BM,BN,BK,TM,TN><<<gridDim4, blockDim4>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);
    cudaEventElapsedTime(&time, startEvent, endEvent);
    printf("2D Block Tiling Matrix Multiplication Time: %f ms | GFLOPS: %.1f |", time, (flops/1e9)/(time));
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    compare_matrices(h_C_naive, h_C, M, N);
    
    //print_matrix(h_C, M, N);

    // ===== Vectorized block tiling matrix multiplication kernel launch ==== //
    time = 0.0f;
    cudaEventRecord(startEvent, 0);
    vectorize_matmul<BM,BN,BK,TM,TN><<<gridDim4, blockDim4>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);
    cudaEventElapsedTime(&time, startEvent, endEvent);
    printf("Vectorized Block Tiling Matrix Multiplication Time: %f ms | GFLOPS: %.1f |", time, (flops/1e9)/(time));
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);   
    compare_matrices(h_C_naive, h_C, M, N);
    //print_matrix(h_C, M, N);


    // Free resources
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);


}