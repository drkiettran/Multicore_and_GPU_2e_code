/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : June 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP device implementation of vector operation a*(a+b)
 To build use  : g++ -fopenmp linear_clause2.cpp -o linear_clause2
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main ()
{
  int i = 1, j = 1;
#pragma omp for simd linear(i) linear(j:2)
  for (int k = 0; k < 4; k++)
    printf ("SIMD %i %i %i\n", k, i, j);

  printf ("POST LOOP %i %i\n", i, j);

#pragma omp parallel for linear(i) linear(j:2)
  for (int k = 0; k < 4; k++)
    printf ("FOR %i %i %i\n", k, i, j);

  printf ("POST LOOP %i %i\n", i, j);

// modified simd loop to exhibit the same behavior
  i = 1, j = 1;
#pragma omp for simd linear(i) linear(j:2)
  for (int k = 0; k < 4; k++)
    {
      printf ("SIMD2 %i %i %i\n", k, i, j);
      i++;
      j += 2;
    }
  i--;
  j -= 2;
  printf ("POST LOOP2 %i %i\n", i, j);

  return 0;
}
