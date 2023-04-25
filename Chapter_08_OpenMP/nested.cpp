/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : June 2020
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp nested.cpp -o nested
 ============================================================================
 */
#include <iostream>
#include <stdlib.h>
#include <omp.h>

using namespace std;

void testPar (int curLvl, int maxLvl)
{
  if (curLvl == maxLvl)
    return;
  else
    {
#pragma omp parallel
      {
#pragma omp single
        {
          cout << "Parallel at lvl " << curLvl << " with thread team of " << omp_get_num_threads () << endl;
          testPar (curLvl + 1, maxLvl);
        }
      }

    }
}

int main (int argc, char **argv)
{
  int nestedLvl = atoi (argv[1]);
  omp_set_max_active_levels (nestedLvl);
  testPar (0, nestedLvl + 1);
  return 0;
}
