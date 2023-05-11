#pragma once

#include "Parameters.h"

void MatMatMultiply(const float (&A)[MATRIX_SIZE_M][MATRIX_SIZE_N],
    const float (&B)[MATRIX_SIZE_N][MATRIX_SIZE_K], float (&C)[MATRIX_SIZE_M][MATRIX_SIZE_K], int threads);

void MatMatMultiplyReference(const float (&A)[MATRIX_SIZE_M][MATRIX_SIZE_N],
    const float (&B)[MATRIX_SIZE_N][MATRIX_SIZE_K], float (&C)[MATRIX_SIZE_M][MATRIX_SIZE_K], int threads);
