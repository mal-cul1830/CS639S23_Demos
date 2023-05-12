#include "PointwiseOps.h"
#include "Utilities.h"

#ifndef DO_NOT_USE_MKL
//#define DO_NOT_USE_MKL = true
#include <mkl.h>
#endif

void Copy(const float (&x)[XDIM][YDIM][ZDIM], float (&y)[XDIM][YDIM][ZDIM])
{
#ifdef DO_NOT_USE_MKL
#pragma omp parallel for    
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++)
        y[i][j][k] = x[i][j][k];
#else
    cblas_scopy(
        XDIM * YDIM * ZDIM,
        &x[0][0][0],
        1,
        &y[0][0][0],
        1
    );
#endif
}

void Saxpy(const float (&x)[XDIM][YDIM][ZDIM], const float (&y)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM],
    const float scale)
{
    const int N = XDIM;

#ifdef DO_NOT_USE_MKL
#pragma omp parallel for
    for (int i = 0; i < XDIM; i++)
    for (int j = 0; j < YDIM; j++)
    for (int k = 0; k < ZDIM; k++)
        z[i][j][k] = x[i][j][k] * scale + y[i][j][k];
#else
    const long int size = XDIM*YDIM*ZDIM;
    cblas_scopy(
        XDIM * YDIM * ZDIM,
        &y[0][0][0],
        1,
        &z[0][0][0],
        1
    );
    cblas_saxpy(
        XDIM * YDIM * ZDIM, // Length of vectors
        scale,              // Scale factor
        &x[0][0][0],        
        1,                  
        &z[0][0][0],        
        1                   
    );
#endif
}

void Saxpy(const float (&x)[XDIM][YDIM][ZDIM], float (&y)[XDIM][YDIM][ZDIM],
    const float scale)
{

#ifdef DO_NOT_USE_MKL    
    // Just for reference -- implementation without MKL
#pragma omp parallel for
    for (int i = 0; i < XDIM; i++)
    for (int j = 0; j < YDIM; j++)
    for (int k = 0; k < ZDIM; k++)
        y[i][j][k] += x[i][j][k] * scale;
#else
    cblas_saxpy(
        XDIM * YDIM * ZDIM, // Length of vectors
        scale,              // Scale factor
        &x[0][0][0],        // Input vector x, in operation y := x * scale + y
        1,                  // Use step 1 for x
        &y[0][0][0],        // Input/output vector y, in operation y := x * scale + y
        1                   // Use step 2 for y
    );
#endif
}
