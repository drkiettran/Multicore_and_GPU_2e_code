/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.1
 Last modified : November 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc odd.cu -o odd
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <memory>

#define MAXVALUE 10000

using namespace std;

//------------------------------------
void numberGen (int N, int max, int *store)
{
  int i;
  srand (time (0));
  for (i = 0; i < N; i++)
    store[i] = rand () % max;
}

//------------------------------------

__global__ void countOdds (int *d, int N, int *odds)
{
  extern __shared__ int count[];

  int myID = blockIdx.x * blockDim.x + threadIdx.x;
  int localID = threadIdx.x;
  count[localID] = 0;
  if (myID < N)
    count[localID] = (d[myID] % 2);
  __syncthreads ();

  // reduction phase: sum up the block
  int step = 1;
  int otherIdx = localID | step;  
  while ((otherIdx < blockDim.x) && ((localID & step) == 0) )
    {
      count[localID] += count[otherIdx];
      step <<= 1;
      otherIdx = localID | step;  
      __syncthreads ();
    }
    
  // add to global counter
  if (localID == 0)
    atomicAdd (odds, count[0]);
}

//------------------------------------
int sharedSize (int b)
{
  return b * sizeof (int);
}

//------------------------------------

int main (int argc, char **argv)
{
  int N = atoi (argv[1]);

  unique_ptr<int[]> ha; // host (h*) and
  int *da;              // device (d*) pointers
  int *dres;
  int hres; 
  
  ha = make_unique<int[]>(N);

  cudaMalloc ((void **) &da, sizeof (int) * N);
  cudaMalloc ((void **) &dres, sizeof (int) * 1);

  numberGen (N, MAXVALUE, ha.get());

  cudaMemcpy (da, ha.get(), sizeof (int) * N, cudaMemcpyHostToDevice);
  cudaMemset (dres, 0, sizeof (int));

  int blockSize, gridSize;
  cudaOccupancyMaxPotentialBlockSizeVariableSMem (&gridSize, &blockSize, (void *) countOdds, sharedSize, N);

  gridSize = ceil (1.0 * N / blockSize);
  printf ("Grid : %i    Block : %i\n", gridSize, blockSize);
  countOdds <<< gridSize, blockSize, blockSize * sizeof (int) >>> (da, N, dres);

  cudaMemcpy (&hres, dres, sizeof (int), cudaMemcpyDeviceToHost);

  // correctness check
  int oc = 0;
  for (int i = 0; i < N; i++)
    if (ha[i] % 2)
      oc++;

  printf ("%i %i\n", hres, oc);

  cudaFree ((void *) da);
  cudaFree ((void *) dres);
  cudaDeviceReset ();

  return 0;
}
