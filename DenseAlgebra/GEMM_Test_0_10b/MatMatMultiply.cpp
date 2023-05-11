#include "MatMatMultiply.h"
#include "mkl.h"

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

    static constexpr int NBLOCKS_M = MATRIX_SIZE_M/BLOCK_SIZE;
    static constexpr int NBLOCKS_N = MATRIX_SIZE_N/BLOCK_SIZE;
    static constexpr int NBLOCKS_K = MATRIX_SIZE_K/BLOCK_SIZE;

    using blocked_matrix_t = float (&) [NBLOCKS_M][BLOCK_SIZE][NBLOCKS_K][BLOCK_SIZE];
    using const_blocked_matrix_t_1 = const float (&) [NBLOCKS_M][BLOCK_SIZE][NBLOCKS_N][BLOCK_SIZE];
    using const_blocked_matrix_t_2 = const float (&) [NBLOCKS_N][BLOCK_SIZE][NBLOCKS_K][BLOCK_SIZE];

    auto blockA = reinterpret_cast<const_blocked_matrix_t_1>(A[0][0]);
    auto blockB = reinterpret_cast<const_blocked_matrix_t_1>(B[0][0]);
    auto blockC = reinterpret_cast<blocked_matrix_t>(C[0][0]);

#pragma omp parallel for
    for (int bi = 0; bi < NBLOCKS_M; bi++)
    for (int bj = 0; bj < NBLOCKS_K; bj++) {

        for (int ii = 0; ii < BLOCK_SIZE; ii++)
            for (int jj = 0; jj < BLOCK_SIZE; jj++) {
                localC[ii][jj] = 0.;
            }

        for (int bk = 0; bk < NBLOCKS_N; bk++) {

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
        MATRIX_SIZE_K,
        MATRIX_SIZE_N,
        1.,
        &A[0][0],
        MATRIX_SIZE_N,
        &B[0][0],
        MATRIX_SIZE_K,
        0.,
        &C[0][0],
        MATRIX_SIZE_K
    );
}
