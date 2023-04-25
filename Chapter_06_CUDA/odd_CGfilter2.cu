/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2019
 License       : Released under the GNU GPL 3.0
 Description   : Filtering out even numbers from input. Odd numbers preserve their relative order
 To build use  : nvcc -arch=compute_60 -rdc=true odd_CGfilter2.cu -o odd_CGfilter2
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <memory>
#include <assert.h>
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
    store[i] = i;               //rand () % max;
}

//------------------------------------
__global__ void filterOdds (int *src, int *dest, int N, int *oddWarpCount)
{
  int myID = blockIdx.x * blockDim.x + threadIdx.x;
  int warpID = myID / 32;       // global warp index
  int totalWarps = blockDim.x * gridDim.x / 32;
  int *toAddToBlock = oddWarpCount + totalWarps;
  bool isOdd = false;

  // init block offsets
  if (myID < gridDim.x)
    toAddToBlock[myID] = 0;

  //------------------------------------
  // Step 1 : find the number of odds assigned to each warp  
  if (myID < N)
    isOdd = src[myID] % 2;      // false for thread beyond N-1

  if (isOdd)
    {
      coalesced_group active = coalesced_threads ();
      if (active.thread_rank () == 0)
        oddWarpCount[warpID] = active.size ();
    }

  grid_group g = this_grid ();
  g.sync ();

  //------------------------------------
  // Step 2 : calculate the prefix-sum of the counts
  if (myID < totalWarps)
    {
      int val = oddWarpCount[myID];
      coalesced_group cg = coalesced_threads ();

      for (int offset = 1; offset < 32; offset *= 2)
        {
          int tmp = cg.shfl_up (val, offset);
          if (cg.thread_rank () >= offset)      // source exists?
            val += tmp;
        }
      oddWarpCount[myID] = val;
    }
  g.sync ();

  //------------------------------------
  // Step 3 : Find by how much to offset each block
  if (myID < gridDim.x)
    {
      int toAdd = oddWarpCount[myID * 32 + 31];
      for (int i = myID + 1; i < gridDim.x; i++)
        atomicAdd (toAddToBlock + i, toAdd);    // multiple warps may be running
    }
  g.sync ();

  // Adjust the warp offsets accordingly
  if (myID < totalWarps)
    oddWarpCount[myID] += toAddToBlock[myID / 32];

  g.sync ();

  //------------------------------------
  // Step 4 : move data
  if (isOdd)
    {
      coalesced_group active = coalesced_threads ();
      int offset;
      if (warpID == 0)
        offset = active.thread_rank ();
      else
        offset = oddWarpCount[warpID - 1] + active.thread_rank ();
      dest[offset] = src[myID];
    }
}

//------------------------------------
int main (int argc, char **argv)
{
  int N = atoi (argv[1]);

  unique_ptr < int[] > h_a;     // host (h*) and
  unique_ptr < int[] > h_odd;
  int *d_a;                     // device (d*) pointers
  int *d_odd;
  int *d_blockCounts;

  h_a = make_unique < int[] > (N);

  cudaMalloc ((void **) &d_a, sizeof (int) * N);

  numberGen (N, MAXVALUE, h_a.get ());

  cudaMemcpy (d_a, h_a.get (), sizeof (int) * N, cudaMemcpyHostToDevice);

  int blockSize = 1024, gridSize;

  cudaDeviceProp pr;
  cudaGetDeviceProperties (&pr, 0);     // replace 0 with appropriate ID in case of a multi-GPU system
  int SM = pr.multiProcessorCount;
  int numBlocksPerSm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor (&numBlocksPerSm, filterOdds, blockSize, 0);
  
  gridSize = ceil (1.0 * N / blockSize);
  assert(gridSize <= SM * numBlocksPerSm); // make sure we can launch
  
  printf ("Grid : %i    Block : %i\n", gridSize, blockSize);

  int warpPerBlock = blockSize / 32;
  // allocate as many counters as warps and blocks
  cudaMalloc ((void **) &d_blockCounts, sizeof (int) * gridSize * (warpPerBlock + 1));  // one for each warp, plus one for every block
  h_odd = make_unique < int[] > (N);
  cudaMalloc ((void **) &d_odd, sizeof (int) * N);

  void *args[] = { &d_a, &d_odd, &N, &d_blockCounts };
  cudaLaunchCooperativeKernel ((void *) filterOdds, gridSize, blockSize, args, 0);      // Instead of  filterOdds <<< gridSize, blockSize >>> (d_a, N, d_odd, d_blockCounts);

  int toGet;
  cudaMemcpy (&toGet, d_blockCounts + gridSize * warpPerBlock - 1, sizeof (int), cudaMemcpyDeviceToHost);

  cudaMemcpy (h_odd.get (), d_odd, toGet * sizeof (int), cudaMemcpyDeviceToHost);

  printf ("#Odds : %i\n", toGet);
  for (int i = 0; i < toGet; i++)
    printf ("%i ", h_odd[i]);

  printf ("\n");

  cudaFree ((void *) d_a);
  cudaFree ((void *) d_odd);
  cudaFree ((void *) d_blockCounts);
  cudaDeviceReset ();

  return 0;
}
