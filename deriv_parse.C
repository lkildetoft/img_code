#include <math.h>
#include "omp.h"

double derivApprox(double f1, double f2, double h) 
{
    double deriv = (f2-f1)/(2*h);
    return deriv;
}

void fillMask(double *mask, long *mv, long nFrames, long nRows, long nCols, double fps) 
{
    double t_step = 1/fps;
    #pragma omp parallel for
    for (int frameNr = 0; frameNr < (nFrames - 2); frameNr++) 
    {
        long *frame1 = &mv[frameNr];
        long *frame2 = &mv[frameNr+2];
        for (int i = 0; i < nRows; i++) 
        {
            for (int j = 0; j < nCols; j++) 
            {
                mask[i*nRows + j] += derivApprox(
                    frame1[i*nRows + j], frame2[i*nRows + j], t_step);
            };
        };
    };
    #pragma omp parallel for    
    for (int i = 0; i < nRows; i++) 
    {
        for (int j = 0; j < nCols; j++) 
        {
            mask[i*nRows + j] /= nFrames;
        };
    };
}
