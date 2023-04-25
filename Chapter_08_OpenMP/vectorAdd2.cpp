/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : June 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP device implementation of vector operation a*(a+b)
 To build use  : clang++-9 -fopenmp -fopenmp-targets=nvptx64 vectorAdd2.cpp -o vectorAdd2
                 OR
                 g++ -fopenmp -foffload=nvptx-none -fno-stack-protector -foffload=-lm -fno-fast-math -fno-associative-math vectorAdd2.cpp -o vectorAdd2 
 ============================================================================
 */

// d = a*(a+b)

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 3
int main ()
{
  int a[N], b[N], c[N], d[1];

  d[0] = 0;
  for (int i = 0; i < N; i++)
    a[i] = b[i] = i+1;

#pragma target enter data map(to:a,b,d) map(alloc:c)

#pragma omp target teams distribute parallel for
  for (int i = 0; i < N; i++)
    c[i] = a[i] + b[i];

#pragma omp target update from(c)

#pragma omp target exit data map(delete:b)

// #pragma omp target teams num_teams(1)
#pragma omp target teams
  {
#pragma omp distribute parallel for reduction(+:d[0])
    for (int i = 0; i < N; i++)
      d[0] += a[i] * c[i];
  }

#pragma target exit data map(from:d) map(release:a,c)


  for (int i = 0; i < N; i++)
    printf ("%i ", c[i]);
  printf ("\nd:%i\n", d[0]);

  return 0;
}
