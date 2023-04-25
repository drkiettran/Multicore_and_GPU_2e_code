/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.1
 Last modified : Novemberr 2019
 License       : Released under the GNU GPL 3.0
 Description   : Example of MPI_Allgather
 To build use  : mpic++ -std=c++17 allgatherMPI.cpp -o allgatherMPI
 ============================================================================
 */

#include<mpi.h>
#include<iostream>
#include<memory>

const int K = 10;

using namespace std;

//*****************************************
int main (int argc, char **argv)
{
  MPI_Init (&argc, &argv);

  int rank, N;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &N);
  MPI_Status status;

  unique_ptr<double[]> localPart = make_unique<double[]>(K);
  unique_ptr<double[]> allParts = make_unique<double[]>(K * N);

  // test data init.
  for (int i = 0; i < K; i++)
    localPart[i] = rank;

  MPI_Allgather(localPart.get(), K, MPI_DOUBLE, allParts.get(), K, MPI_DOUBLE, MPI_COMM_WORLD);

  // printout/verification step 
  if (rank == 0)
    {
      for (int i = 0; i < K * N; i++)
        cout << allParts[i] << " ";
      cout << endl;
    }

  MPI_Finalize ();
  return 0;
}
