/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : June 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP integration of testf on a GPU
 To build use  : clang++-9 -fopenmp -fopenmp-targets=nvptx64 integration_omp_V5.cpp -o integration_omp_V5
                 OR
                 g++ -fopenmp -foffload=nvptx-none -fno-stack-protector -foffload=-lm -fno-fast-math -fno-associative-math integration_omp_V5.cpp -o integration_omp_V5 
 ============================================================================
 */
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <memory>
#include <omp.h>

using namespace std;

//---------------------------------------
#pragma omp declare target
double testf (double x)
{
  return x * x + 2 * sin (x);
}
#pragma omp end declare target

const int numTeams = 100;
//---------------------------------------
double integrate (double st, double en, int div)
{
  double localRes = 0;
  double step = (en - st) / div;
  unique_ptr < double[] > TR = make_unique < double[] > (numTeams);
  double *teamResults = TR.get ();
#pragma omp target data map(tofrom: teamResults[0:numTeams])
#pragma omp target teams num_teams(numTeams)
  {
    int teamID = omp_get_team_num ();
#pragma omp distribute parallel for reduction(+:teamResults[teamID])
    for (int i = 1; i < div; i++)
      {
        double x = st + i * step;
        teamResults[teamID] += testf (x);
      }
  }
  // consolidate partial results
  for (int i = 0; i < numTeams; i++)
    localRes += teamResults[i];

  localRes += (testf (st) + testf (en)) / 2;
  localRes *= step;
  return localRes;
}
//---------------------------------------
int main (int argc, char *argv[])
{

  if (argc == 1)
    {
      cerr << "Usage " << argv[0] << " start end divisions\n";
      exit (1);
    }
  double start, end;
  int divisions;
  start = atof (argv[1]);
  end = atof (argv[2]);
  divisions = atoi (argv[3]);

  double finalRes = integrate (start, end, divisions);

  cout << finalRes << endl;
  return 0;
}
