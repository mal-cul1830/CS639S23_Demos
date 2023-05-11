#include "MatMatMultiply.h"
#include "Timer.h"
#include "Utilities.h"

#include <iostream>
#include <iomanip>
#include <limits.h>

int main(int argc, char *argv[])
{
    float *Araw = static_cast<float*>( AlignedAllocate( MATRIX_SIZE_M * MATRIX_SIZE_N * sizeof(float), 64 ) );
    float *Braw = static_cast<float*>( AlignedAllocate( MATRIX_SIZE_N * MATRIX_SIZE_K * sizeof(float), 64 ) );
    float *Craw = static_cast<float*>( AlignedAllocate( MATRIX_SIZE_M * MATRIX_SIZE_K * sizeof(float), 64 ) );
    float *referenceCraw = static_cast<float*>( AlignedAllocate( MATRIX_SIZE_M * MATRIX_SIZE_K * sizeof(float), 64 ) );

    using matrix_t_1 = float (&) [MATRIX_SIZE_M][MATRIX_SIZE_N];
    using matrix_t_2 = float (&) [MATRIX_SIZE_N][MATRIX_SIZE_K];
    using matrix_t_3 = float (&) [MATRIX_SIZE_M][MATRIX_SIZE_K];

    matrix_t_1 A = reinterpret_cast<matrix_t_1>(*Araw);
    matrix_t_2 B = reinterpret_cast<matrix_t_2>(*Braw);
    matrix_t_3 C = reinterpret_cast<matrix_t_3>(*Craw);
    matrix_t_3 referenceC = reinterpret_cast<matrix_t_3>(*referenceCraw);

    InitializeMatrices(A, B);
    Timer timer;

    int min_threads = INT_MAX;
    int min_time = INT_MAX;

    for(int threads = 1; threads <= 15; threads += 5){
        std::cout<<"Running with "<<threads<<" threads. ";

        // Correctness test
        std::cout << "Running candidate kernel for correctness test ... " << std::flush;
        timer.Start();
        MatMatMultiply(A, B, C, threads);
        timer.Stop("Elapsed time : ");
        
        std::cout << "Running reference kernel for correctness test ... " << std::flush;
        timer.Start();
        MatMatMultiplyReference(A, B, referenceC, threads);
        timer.Stop("Elapsed time : ");

        float discrepancy = MatrixMaxDifference(C, referenceC);
        std::cout << "Discrepancy between two methods : " << discrepancy << std::endl;

        float sum = 0;

        for(int test = 1; test <= 20; test++)
        {
            std::cout << "Running kernel for performance run #" << std::setw(2) << test << " ... ";
            timer.Start();
            MatMatMultiply(A, B, C, threads);
            sum += timer.Stop("");//"Elapsed time : ");
        }

        min_threads = ((sum/20) < min_time)?threads:min_threads;
        min_time = ((sum/20) < min_time)?(sum/20):min_time;
        std::cout<<"Avg. Time : "<<sum/20<<"ms. \n";
    }
    
    return 0;
}
