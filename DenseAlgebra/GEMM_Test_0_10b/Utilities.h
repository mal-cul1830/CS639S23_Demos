#pragma once

#include "Parameters.h"

#include <cstdlib>

void* AlignedAllocate(const std::size_t size, const std::size_t alignment);
void InitializeMatrices(float (&A)[MATRIX_SIZE_M][MATRIX_SIZE_N],float (&B)[MATRIX_SIZE_N][MATRIX_SIZE_K]);
float MatrixMaxDifference(const float (&A)[MATRIX_SIZE_M][MATRIX_SIZE_K],const float (&B)[MATRIX_SIZE_M][MATRIX_SIZE_K]);
