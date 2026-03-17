/******************************************************************************
* FILE: omp_reduction.c
* DESCRIPTION:
*   OpenMP Example - Combined Parallel Loop Reduction - C/C++ Version
*   This example demonstrates a sum reduction within a combined parallel loop
*   construct.  Notice that default data element scoping is assumed - there
*   are no clauses specifying shared or private variables.  OpenMP will 
*   automatically make loop index variables private within team threads, and
*   global variables shared.
* AUTHOR: Blaise Barney  5/99
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
    int   i, n;
    float a[100], b[100], sum; 

    /* Some initializations */
    n = 100;
    for (i=0; i < n; i++)
    a[i] = b[i] = i * 1.0;
    sum = 0.0;

    // 'reduction(+:sum)' handles the thread-safety automatically
    // Under the hood, OpenMP creates a private copy of 'sum' for each thread, initializes it to zero
    // Each thread performs its portion of the loop, updating its private 'sum' variable
    // OpenMP combines all the private 'sum' variables into a single value using the specified reduction operation 
    // Reduction is done using a tree-based approach, where pairs of private sums are combined in parallel until a single total sum is obtained.
    #pragma omp parallel for reduction(+:sum)
    for (i=0; i < n; i++)
        sum = sum + (a[i] * b[i]);

    printf("   Sum = %f\n",sum);

}
