#include "MatMatMultiply.h"
#include "mkl.h"
#include <omp.h>

alignas(64) float localA[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) float localB[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) float localC[BLOCK_SIZE][BLOCK_SIZE];

#pragma omp threadprivate(localA, localB, localC)

void MatMatMultiply(const float (&A)[MATRIX_SIZE_M][MATRIX_SIZE_N],
    const float (&B)[MATRIX_SIZE_N][MATRIX_SIZE_K], float (&C)[MATRIX_SIZE_M][MATRIX_SIZE_K], 
    int threads) //adding threads as a parameter
{
    omp_set_dynamic(0); 
    omp_set_num_threads(threads);

    static constexpr int NBLOCKS = 4096;
    static constexpr int BLOCK_SIZE = MATRIX_SIZE_M/FACTOR;

    using blocked_matrix_t = float (&) [NBLOCKS][BLOCK_SIZE][NBLOCKS][BLOCK_SIZE];
    using const_blocked_matrix_t_1 = const float (&) [NBLOCKS][BLOCK_SIZE][NBLOCKS][BLOCK_SIZE];
    using const_blocked_matrix_t_2 = const float (&) [NBLOCKS][BLOCK_SIZE][NBLOCKS][BLOCK_SIZE];

    auto blockA = reinterpret_cast<const_blocked_matrix_t>(A[0][0]);
    auto blockB = reinterpret_cast<const_blocked_matrix_t>(B[0][0]);
    auto blockC = reinterpret_cast<blocked_matrix_t>(C[0][0]);

#pragma omp parallel for
    for (int bi = 0; bi < NBLOCKS; bi++)
    for (int bj = 0; bj < NBLOCKS; bj++) {
        
        for (int ii = 0; ii < BLOCK_SIZE; ii++)
            for (int jj = 0; jj < BLOCK_SIZE; jj++) {
                localC[ii][jj] = 0.;
            }

        for (int bk = 0; bk < BLOCK_SIZE; bk++) { 

            for (int ii = 0; ii < BLOCK_SIZE; ii++)
            for (int jj = 0; jj < BLOCK_SIZE; jj++) {
                localA[ii][jj] = blockA[bi][ii][bk][jj];
            }

            for (int ii = 0; ii < BLOCK_SIZE; ii++)
            for (int jj = 0; jj < BLOCK_SIZE; jj++) {
                localB[ii][jj] = blockB[bk][ii][bj][jj];
            }

            for (int ii = 0; ii < BLOCK_SIZE; ii++)
                for (int kk = 0; kk < BLOCK_SIZE; kk++)
#pragma omp simd
                    for (int jj = 0; jj < BLOCK_SIZE; jj++)
                    localC[ii][jj] += localA[ii][kk] * localB[kk][jj];
        }

        for (int ii = 0; ii < BLOCK_SIZE; ii++)
        for (int jj = 0; jj < BLOCK_SIZE; jj++)                
            blockC[bi][ii][bj][jj] = localC[ii][jj];
    }
}

void MatMatMultiplyReference(const float (&A)[MATRIX_SIZE_M][MATRIX_SIZE_N],
    const float (&B)[MATRIX_SIZE_N][MATRIX_SIZE_K], float (&C)[MATRIX_SIZE_M][MATRIX_SIZE_K], int threads)
{
    omp_set_dynamic(0); 
    omp_set_num_threads(threads);
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        MATRIX_SIZE_M,
        MATRIX_SIZE_N,
        MATRIX_SIZE_K,
        1.,
        &A[0][0],
        MATRIX_SIZE_M,
        &B[0][0],
        MATRIX_SIZE_N,
        0.,
        &C[0][0],
        MATRIX_SIZE_K
    );
}
