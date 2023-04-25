/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : November 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : qmake mpiAndThreads.pro; make
 ============================================================================
 */
#include<mpi.h>
#include<iostream>
#include<unistd.h>
#include<thread>
#include<memory>

using namespace std;
//---------------------------------------
int main (int argc, char **argv)
{
  MPI_Init (&argc, &argv);

  int rank, N;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &N);

  int numThreads = sysconf (_SC_NPROCESSORS_ONLN);
  unique_ptr<thread> thr[numThreads];
  for (int i = 0; i < numThreads; i++)
    {
      thr[i] = make_unique<thread>([=](){
          cout << "Thread " << i << " is running on process " << rank << "\n";
                              }
                           );
    }

  for (int i = 0; i < numThreads; i++)
    thr[i]->join();

  MPI_Finalize ();
  return 0;
}
