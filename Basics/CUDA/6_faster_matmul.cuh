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

    // CHANGE: correct block size assertion for 1D tiling
    assert((BM / TM) * BN == blockDim.x);

    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;    

    float threadResults[TM] = {0.0f};

    for (uint bk = 0; bk < K; bk += BK) {

        Asub[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bsub[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint j = 0; j < BK; ++j)
        {
            float tempB = Bsub[j * BN + threadCol];
            for (uint t = 0; t < TM; ++t)
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

    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN; 

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    // register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    const uint strideA = blockDim.x / BK; // Rows within BM*BK tile
    const uint strideB = blockDim.x / BN; // Rows within BK*BN tile

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        
        // Loads elements along BM
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            Asub[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
        }
        
        // Loads elements along BN
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
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
    int errors = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float diff = fabs(mat1[i * cols + j] - mat2[i * cols + j]);
            if (diff > epsilon) {
                errors++;
                if (errors <= 3) printf("  Differ at (%d, %d): %f vs %f\n", i, j, mat1[i * cols + j], mat2[i * cols + j]);
            }
        }
    }
    if (errors == 0) {
        printf("  ✓ Correct\n");
    } else {
        printf("  ✗ %d mismatches\n", errors);
    }
}