#include "aitken.h"

double aitken(double *series, int len)
{
    for(int k = 0; k < len - 2; k++)
    {
        double p0 = series[k];
        double p1 = series[k + 1];
        double p2 = series[k + 2];
        
        series[k] = p2 - (p2 - p1) * (p2 - p1) / (p2 - 2 * p1 + p0);
    }
}

