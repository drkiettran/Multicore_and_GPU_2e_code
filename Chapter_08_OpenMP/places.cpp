/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : July 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP thread affinity display
 To build use  : g++ -fopenmp places.cpp -o places
 ============================================================================
 */
#include<iostream>
#include<omp.h>

using namespace std;

int main ()
{
  int nPlaces = omp_get_num_places ();
  cout << "Places : " << nPlaces << endl;
  for (int i = 0; i < nPlaces; i++)
    {
      int numProcs = omp_get_place_num_procs (i);
      cout << "\tPlace " << i << " with " << numProcs << endl;
      int IDs[numProcs];
      omp_get_place_proc_ids (i, IDs);
      for (int j = 0; j < numProcs; j++)
        cout << "\t\tCPU with ID : " << IDs[j] << endl;
    }
// #pragma omp parallel proc_bind(spread)
#pragma omp parallel
  {
// #pragma omp single
//     {
//       cout << "Single thread running at : " << omp_get_place_num () << endl;
//       cout << "\t with place nums " << omp_get_partition_num_places () << endl;
//       int pIDs[omp_get_partition_num_places ()];
//       for (int k = 0; k < omp_get_partition_num_places (); k++)
//         cout << "\t\t" << pIDs[k] << endl;
// //  cout << "Affinity " <<
//       omp_display_affinity (NULL);
//     }

#pragma omp for
    for (int l = 0; l < omp_get_num_threads (); l++)
      printf ("Thread %i at place %i\n", omp_get_thread_num (), omp_get_place_num ());
  }

  return 0;
}
