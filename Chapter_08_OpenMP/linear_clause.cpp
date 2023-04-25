/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : June 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP integration of testf
 To build use  : g++ -fopenmp linear_clause.cpp -o linear_clause
 ============================================================================
 */
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;

const int N = 10;
//---------------------------------------
int main (int argc, char *argv[])
{
  int i = 0, j = 1;
  float x[N];

  for (int k = 0; k < N; k++)
    {
      cout << k << ": " << i << " " << j << endl;
      x[k] = i + j * j;
      i++;
      j += 2;
    }
  for (int k = 0; k < N; k++)
    cout << x[k] << " ";
  cout << endl << "===========\n";

  i = 0;
  j = 1;
#pragma omp parallel for linear(i) linear(j:2)
  for (int k = 0; k < N; k++)
    {
      x[k] = i + j * j;
#pragma omp critical
      cout << k << ": " << i << " " << j << endl;
    }

  for (int k = 0; k < N; k++)
    cout << x[k] << " ";
  cout << endl;
  return 0;
}
