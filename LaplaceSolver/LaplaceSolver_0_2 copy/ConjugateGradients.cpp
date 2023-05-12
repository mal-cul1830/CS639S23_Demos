#include "Laplacian.h"
#include "Parameters.h"
#include "PointwiseOps.h"
#include "Reductions.h"
#include "Utilities.h"
#include "Timer.h"

#include <iostream>

extern Timer timerInt;
extern int iterations;

void ConjugateGradients(
    float (&x)[XDIM][YDIM][ZDIM],
    const float (&f)[XDIM][YDIM][ZDIM],
    float (&p)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM],
    float (&t)[6][17],
    const bool writeIterations)
{
    // Algorithm : Line 2
    timerInt.Restart(); ComputeLaplacian(x, z); t[0][2] += timerInt.Pause();
    timerInt.Restart(); Saxpy(z, f, r, -1); t[1][2] += timerInt.Pause();
    timerInt.Restart(); float nu = Norm(r); t[2][2] += timerInt.Pause();

    // Algorithm : Line 3
    if (nu < nuMax) return;
        
    // Algorithm : Line 4
    timerInt.Restart(); Copy(r, p); t[3][4] += timerInt.Pause();
    timerInt.Restart(); float rho=InnerProduct(p, r); t[4][4] += timerInt.Pause();
        
    // Beginning of loop from Line 5
    for(int k=0;;k++)
    {
        std::cout << "Residual norm (nu) after " << k << " iterations = " << nu << std::endl;
        ++iterations;
        // Algorithm : Line 6
        
        //timerInt.Restart(); ComputeLaplacian(p, z); t[0][6] += timerInt.Pause();
        //timerInt.Restart(); float sigma=InnerProduct(p, z); t[4][6] += timerInt.Pause();
        

        timerInt.Restart(); float sigma = Compressed(p, z); t[5][6] += timerInt.Pause();

        // Algorithm : Line 7
        float alpha=rho/sigma;

        // Algorithm : Line 8
        timerInt.Restart(); Saxpy(z, r, r, -alpha); t[1][8] += timerInt.Pause();
        timerInt.Restart(); nu=Norm(r); t[2][8] += timerInt.Pause();
        
        // Algorithm : Lines 9-12
        if (nu < nuMax || k == kMax) {
            timerInt.Restart(); Saxpy(p, x, x, alpha); t[1][9] += timerInt.Pause();
            std::cout << "Conjugate Gradients terminated after " << k << " iterations; residual norm (nu) = " << nu << std::endl;
            //timerInt.Restart(); if (writeIterations) WriteAsImage("x", x, k, 0, 127); t[5][9] += timerInt.Pause();
            return;
        }
            
        // Algorithm : Line 13
        timerInt.Restart(); Copy(r, z); t[3][13] += timerInt.Pause();
        timerInt.Restart(); float rho_new = InnerProduct(z, r); t[4][13] += timerInt.Pause();

        // Algorithm : Line 14
        float beta = rho_new/rho;

        // Algorithm : Line 15
        rho=rho_new;

        // Algorithm : Line 16
        timerInt.Restart(); Saxpy(p, x, x, alpha); t[1][16] += timerInt.Pause();
        timerInt.Restart(); Saxpy(p, r, p, beta); t[1][16] += timerInt.Pause();

        timerInt.Restart(); if (writeIterations) WriteAsImage("x", x, k, 0, 127); t[5][9] += timerInt.Pause();
    }
}
