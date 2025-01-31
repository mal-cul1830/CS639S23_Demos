#include "ConjugateGradients.h"
#include "Laplacian.h"
#include "Timer.h"
#include "Utilities.h"

Timer timerInt;
Timer timerTotal;
int iterations;

int main(int argc, char *argv[])
{
    using array_t = float (&) [XDIM][YDIM][ZDIM];

    float *xRaw = new float [XDIM*YDIM*ZDIM];
    float *fRaw = new float [XDIM*YDIM*ZDIM];
    float *pRaw = new float [XDIM*YDIM*ZDIM];
    float *rRaw = new float [XDIM*YDIM*ZDIM];
    float *zRaw = new float [XDIM*YDIM*ZDIM];
    
    array_t x = reinterpret_cast<array_t>(*xRaw);
    array_t f = reinterpret_cast<array_t>(*fRaw);
    array_t p = reinterpret_cast<array_t>(*pRaw);
    array_t r = reinterpret_cast<array_t>(*rRaw);
    array_t z = reinterpret_cast<array_t>(*zRaw);
    
    CSRMatrix matrix;
    using array_t_1 = float (&) [6][17];
    float *tRaw = new float [6*17];
    array_t_1 t = reinterpret_cast<array_t_1>(*tRaw);

    // Initialization
    {
        Timer timer;
        timer.Start();
        InitializeProblem(x, f);
        matrix = BuildLaplacianMatrix(); // This takes a while ...
        timer.Stop("Initialization : ");
        iterations = 0;
    }

    // Call Conjugate Gradients algorithm
    timerTotal.Reset();
    {	
        timerTotal.Start();
        ConjugateGradients(matrix, x, f, p, r, z, t, false);
        timerTotal.Stop("Total Time for Conjugate Gradients : ");
        const char* kernel[6]= {"ComputeLaplacian", "Saxpy", "Norm",
        "Copy", "InnerProduct", "Compressed"};

        for(int i = 0; i < 6; ++i) {
            for(int j = 1; j <= 16; ++j){
                if(t[i][j])
                    std::cout << kernel[i] << ", on Line "<<j<<" time : " << 
                    (t[i][j])/((iterations-1)*int(j > 5) + 1*int(j < 13)) <<" ms" <<std::endl;
            }
        }

        for(int i = 0; i < 6; ++i) {
            float sum = 0;
            int k = 0;
            for(int j = 5; j <= 16; ++j){
                if(t[i][j] > 0){
                    sum += t[i][j];
                    k += 1;
                }
            }
            if(sum > 0)
                std::cout << "Average Time for " <<kernel[i]<< sum/k <<"\n";
        }
    }

    return 0;
}
