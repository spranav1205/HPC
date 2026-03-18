#include "aitken.h"
#include <stdio.h>
#include <math.h>

double log_series(double x, int n)
{
    double sum = 0.0;
    for (int k = 1; k <= n; k++)
    {
        sum += (k % 2 == 1 ? 1 : -1) * (x - 1) / k;
    }
    return sum;
}

double pi_series(int n)
{
    double sum = 0.0;
    for (int k = 0; k < n; k++)
    {
        sum += (k % 2 == 0 ? 1 : -1) / (2.0 * k + 1);
    }
    return 4 * sum;
}

#define pi 3.14159265358979323846

int accurate_digits(double prediction, double reference) {
    double diff = prediction - reference;
    if (diff < 0) diff = -diff; // Manual fabs()

    if (diff == 0.0) return 16;
    if (diff >= 1.0) return 0;

    int count = 0;
    // Keep multiplying by 10 until we reach the units place
    while (diff < 1.0 && count < 16) {
        diff *= 10.0;
        count++;
    }

    // Adjusting by 1 because the first digit is the 0.X place
    return count - 1;
}

double log_series_aitken(double *series, int n) {
    while (n > 2) {
        aitken(series, n);
        n -= 2;
    }
    return series[0];
}

double pi_series_aitken(double *series, int n)
{
    while(n > 2) {
        aitken(series, n);
        n -= 2;
    }
    return series[0];
}

int main()
{
    double x = 1.5;
    int n = 16;

    double series[16];
    for (int i = 0; i < n; i++)
    {
        series[i] = log_series(x, i + 1);
    }

    double log_approx = log_series(x, n);
    double log_aitken = log_series_aitken(series, n);
    printf("Log approximation: %f, Accurate digits: %d, Aitken's method: %f, Accurate digits: %d\n",
           log_approx, accurate_digits(log_approx, log(x)), log_aitken, accurate_digits(log_aitken, log(x)));

    for (int i = 0; i < n; i++)
    {
        series[i] = pi_series(i + 1);
    }

    double pi_approx = pi_series(n);
    double pi_aitken = pi_series_aitken(series, n);
    printf("Pi approximation: %f, Accurate digits: %d, Aitken's method: %f, Accurate digits: %d\n",
           pi_approx, accurate_digits(pi_approx, pi), pi_aitken, accurate_digits(pi_aitken, pi));

    return 0;
}