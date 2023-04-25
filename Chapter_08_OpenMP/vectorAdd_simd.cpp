/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : June 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP adding two vectors on a device in chunks, using array sections
 To build use  : g++ -fopenmp vectorAdd_simd.cpp -o vectorAdd_simd
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <memory>

using namespace std;

#pragma omp declare simd uniform(a)
float saxpy (float a, float x, float y)
{
  return a * x + y;
}

int main (int argc, char **argv)
{
  int N = atoi (argv[1]);
  int useSIMD = atoi (argv[2]);

  unique_ptr < float[] > x = make_unique < float[] > (N);
  unique_ptr < float[] > y = make_unique < float[] > (N);
  unique_ptr < float[] > z = make_unique < float[] > (N);
#pragma omp parallel for simd if(useSIMD>0)
  for (int i = 0; i < N; i++)
    x[i] = y[i] = i;

  float a = 1.0;
#pragma omp parallel for simd if(useSIMD>0)
  for (int i = 0; i < N; i++)
    z[i] = saxpy (a, x[i], y[i]);

  for (int i = 0; i < min(N,100); i++)
    printf ("%f ", z[i]);
  printf ("\n");

  return 0;
}
