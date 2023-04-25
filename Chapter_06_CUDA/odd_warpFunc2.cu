/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc  odd_warpFunc2.cu -o odd_warpFunc2
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <memory>
#include <cooperative_groups.h>

#define MAXVALUE 10000

using namespace std;
using namespace cooperative_groups;

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
  int warpID = localID / 32;
  count[localID] = 0;
  __syncthreads ();

  // Phase 1 : warp calculation   
  if (myID < N)
    {
      if (d[myID] % 2)
        {
          unsigned mask = __activemask ();
          int tmp = __popc (mask);      // count the bits
          int lsb = __ffs (mask);       // find the first lane to be in this block
          if (localID % 32 == lsb - 1)
            count[warpID] = tmp;
        }
    }

// // alternative for phase 1
//   int predicate = 0;
//   if (myID < N)
//     predicate = d[myID] % 2;
//   unsigned mask = 0xffffffff;
//   unsigned active = __ballot_sync (mask, predicate);
//   int tmp = __popc (active);
//   int lsb = __ffs (mask);
//   if (localID % 32 == lsb - 1)
//     count[warpID] = tmp;

  __syncthreads ();

  // Phase 2 : partial results consolidation
  if (warpID == 0)
    {
      unsigned mask = 0xffffffff;
      int val = count[localID];
      for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync (mask, val, offset);

      if (localID == 0)         // lane 0 in all warps
        atomicAdd (odds, val);
    }
}

//------------------------------------
int sharedSize (int b)
{
  return (b / 32 + 1) * sizeof (int);
}

//------------------------------------

int main (int argc, char **argv)
{
  int N = atoi (argv[1]);

  unique_ptr < int[] > ha;      // host (h*) and
  int *da;                      // device (d*) pointers
  int *dres;
  int hres;

  ha = make_unique < int[] > (N);

  cudaMalloc ((void **) &da, sizeof (int) * N);
  cudaMalloc ((void **) &dres, sizeof (int) * 1);

  numberGen (N, MAXVALUE, ha.get ());

  cudaMemcpy (da, ha.get (), sizeof (int) * N, cudaMemcpyHostToDevice);
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
