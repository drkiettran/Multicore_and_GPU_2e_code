/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : June 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP adding two vectors on a device in chunks, using array sections
 To build use  : clang++-9 -fopenmp -fopenmp-targets=nvptx64 vectorAdd3.cpp -o vectorAdd3
                 OR
                 g++ -fopenmp -foffload=nvptx-none -fno-stack-protector -foffload=-lm -fno-fast-math -fno-associative-math vectorAdd3.cpp -o vectorAdd3 
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int N=200000000;
const int chunk=10000;

int a[N], b[N], c[N];

int main ()
{

  for(int i=0;i<N;i++)
     a[i] = b[i] = i;

for(int off=0; off<N; off+=chunk)  
#pragma omp target teams distribute parallel for map(to:a[off:chunk],b[off:chunk]) map(from:c[off:chunk])  
  for(int i=0;i<chunk;i++)
     c[off+i] = a[off+i]+b[off+i];

 for(int i=0;i<100;i++)
    printf("%i ", c[i]);
  printf("\n");
 
  return 0;
}
