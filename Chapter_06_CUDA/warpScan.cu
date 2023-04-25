/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc warpScan.cu -o warpScan
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
  for (i = 0; i < N; i++)
    store[i] = 1;
}

//------------------------------------

__global__ void scan (int *d, int N, int *res)
{
  extern __shared__ int shm[];
  __shared__ int *toadd;     // what to add to each warp partial results
  toadd = shm + blockDim.x;  // find beginning of toadd array

  int myID = threadIdx.x;
  int warpID = myID / 32;
  if (myID % 32 == 0)
    toadd[warpID] = 0;

  shm[myID] = 0;
  if (myID < N)
    shm[myID] = d[myID];
  __syncthreads ();

  // Scan phase 1 : partial warp results
  unsigned mask = 0xffffffff;
  int val = shm[myID];
  for (int offset = 1; offset < 32; offset *= 2)
    {
      int tmp = __shfl_up_sync (mask, val, offset);
      if (myID % 32 - offset >= 0)
        val += tmp;
    }
  shm[myID] = val;
  __syncthreads ();

  // Scan phase 2 : get lane 0 in each warp to find what to add to each warp partial results
  if (myID % 32 == 0)
    {
      for (int i = myID + 32; i < blockDim.x; i += 32)
        atomicAdd (toadd + (i / 32), shm[myID + 31]);   // multiple warps may be running
    }
  __syncthreads ();

  // Scan phase 3 : complete calculation
  if (warpID > 0)
    shm[myID] += toadd[warpID];

  // Store in global memory   
  if (myID < N)
    res[myID] = shm[myID];
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

  unique_ptr < int[] > ha;      // host (h*) and
  unique_ptr < int[] > hres;
  int *da;                      // device (d*) pointers
  int *dres;

  ha = make_unique < int[] > (N);
  hres = make_unique < int[] > (N);

  cudaMalloc ((void **) &da, sizeof (int) * N);
  cudaMalloc ((void **) &dres, sizeof (int) * N);

  numberGen (N, MAXVALUE, ha.get ());

  cudaMemcpy (da, ha.get (), sizeof (int) * N, cudaMemcpyHostToDevice);

  int blockSize = N;
  int gridSize = 1;

  scan <<< gridSize, blockSize, (blockSize + (blockSize/32+1)) * sizeof (int) >>> (da, N, dres);

  cudaMemcpy (hres.get (), dres, sizeof (int) * N, cudaMemcpyDeviceToHost);

  // correctness check
  for (int i = 0; i < N; i++)
    printf ("%i ", hres[i]);

  printf ("\n");

  cudaFree ((void *) da);
  cudaFree ((void *) dres);
  cudaDeviceReset ();

  return 0;
}
