/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.1
 Last modified : November 2019
 License       : Released under the GNU GPL 3.0
 Description   : All-gather implementation.
                 Number of processes must be a power-of-2
 To build use  : mpic++ allgather.cpp -o allgather -std=c++17
 ============================================================================
 */
#include<mpi.h>
#include<memory>
#include<iostream>

const int K = 10;
const int ALLGATHERTAG = 0;

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
  unique_ptr<double[]> allParts = make_unique<double[]>(K*N);

  for (int i = 0; i < K; i++)
    localPart[i] = rank;
  int bitMask = 1;
  int acquiredCount = K;
  int acquiredStart = rank * K;

  // copy local data to array that will hold the complete data
  memcpy (&allParts[rank * K], &localPart[0], sizeof (double) * K);
  while (bitMask < N)
    {
      int otherPartyID = rank ^ bitMask;
      if ((rank & bitMask) == 0)
        {
          MPI_Send (&allParts[acquiredStart], acquiredCount, MPI_DOUBLE, otherPartyID, ALLGATHERTAG, MPI_COMM_WORLD);
          MPI_Recv (&allParts[acquiredStart + acquiredCount], acquiredCount, MPI_DOUBLE, otherPartyID, ALLGATHERTAG, MPI_COMM_WORLD, &status);
          acquiredCount *= 2;
        }
      else
        {
          MPI_Recv (&allParts[acquiredStart - acquiredCount], acquiredCount, MPI_DOUBLE, otherPartyID, ALLGATHERTAG, MPI_COMM_WORLD, &status);
          MPI_Send (&allParts[acquiredStart], acquiredCount, MPI_DOUBLE, otherPartyID, ALLGATHERTAG, MPI_COMM_WORLD);
          acquiredStart -= acquiredCount;
          acquiredCount *= 2;
        }
      bitMask <<= 1;
    }

  if (rank == 0)
    {
      for (int i = 0; i < K * N; i++)
        cout << allParts[i] << " ";
      cout << endl;
    }


  MPI_Finalize ();
  return 0;
}
