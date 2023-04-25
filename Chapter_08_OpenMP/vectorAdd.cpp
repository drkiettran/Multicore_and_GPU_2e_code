/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : June 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP adding two vectors on a device
 To build use  : clang++-9 -fopenmp -fopenmp-targets=nvptx64 vectorAdd.cpp -o vectorAdd
                 OR
                 g++ -fopenmp -foffload=nvptx-none -fno-stack-protector -foffload=-lm -fno-fast-math -fno-associative-math vectorAdd.cpp -o vectorAdd 
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 100
int main ()
{
  int a[N], b[N], c[N];

  for(int i=0;i<N;i++)
     a[i] = b[i] = i;
  
// #pragma target data map(to:a,b) map(from:c)  
// #pragma omp target teams distribute parallel for
#pragma omp target teams distribute parallel for map(to:a,b) map(from:c)  
  for(int i=0;i<N;i++)
     c[i] = a[i]+b[i];

 for(int i=0;i<N;i++)
    printf("%i ", c[i]);
  printf("\n");
 
  return 0;
}
