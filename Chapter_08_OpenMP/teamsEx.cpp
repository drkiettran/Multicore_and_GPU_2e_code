/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : June 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP integration of testf
 To build use  : clang++-9 -fopenmp -fopenmp-targets=nvptx64  teamsEx.cpp -o teamsEx
                 OR
                 g++ -fopenmp -fno-stack-protector teamsEx.cpp -o teamsEx -foffload=-lm 
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 10
int main ()
{
  int a[N][N], b[N][N], c[N][N];
  printf ("%i\n", omp_get_initial_device ());
  printf ("%i\n", omp_get_num_devices ());

#pragma omp target
#pragma omp teams
  {
#pragma omp distribute
    for (int i = 0; i < N; i++)
      {
        printf ("Team %i  (%i)\n", omp_get_team_num (), i);
#pragma omp parallel for
        for (int j = 0; j < N; j++)
          for (int k = 0; k < N; k++)
            {
              c[i][j] += a[i][k] * b[k][j];
              printf ("Team %i   Thread %i (%i,%i,%i)\n", omp_get_team_num (), omp_get_thread_num (), i, j, k);
            }
      }
  }

  return 0;
}
