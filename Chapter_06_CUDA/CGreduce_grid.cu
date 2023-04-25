/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : August 2020
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc CGreduce_grid.cu -o CGreduce_grid
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <memory>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_pipeline.h>

#define MAXVALUE 10000

using namespace std;
using namespace cooperative_groups;
using namespace nvcuda::experimental;

#define CUDA_CHECK_RETURN(value) {           \
    cudaError_t _m_cudaStat = value;         \
    if (_m_cudaStat != cudaSuccess) {        \
         fprintf(stderr, "Error %s at line %d in file %s\n",              \
                 cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);    \
         exit(1);                                                         \
       } }

const int BLKSIZE = 256;

//------------------------------------
void numberGen (int N, int max, int *store)
{
  int i;
  srand (time (0));
  for (i = 0; i < N; i++)
    store[i] = i;               //rand () % max;
}

//------------------------------------
__global__ void countOdds (int *src, int N, int *odd, int *perBlock)
{
  int oddCount = 0;
  __shared__ int cache[32];     // maximum possible warps 

  thread_block blk = this_thread_block ();
  grid_group grd = this_grid ();

  int myID = grd.thread_rank ();
  int localID = blk.thread_rank ();
  int totalThr = grd.size ();

  // calculate local result
  for (int i = myID; i < N; i += totalThr)
    oddCount += (src[i] % 2);

  //------------
  // Step-1 : reduction on a per-warp basis (tile is 32 threads)
  thread_block_tile < 32 > tile = tiled_partition < 32 > (blk);
  oddCount = reduce (tile, oddCount, cooperative_groups::plus < int >());

  // store the tile results in shared memory
  if (tile.thread_rank () == 0)
      cache[tile.meta_group_rank ()] = oddCount;
  blk.sync ();

  //------------
  // Step-2 : reduce the block results into one. First warp in block employed for this
  if (tile.meta_group_rank () == 0 && localID < (blk.size () + warpSize - 1) / warpSize)
    {
      coalesced_group activeWarps = coalesced_threads ();
      oddCount = cache[localID];
      oddCount = reduce (activeWarps, oddCount, cooperative_groups::plus < int >());
 
      // store block result in global memory
      if (activeWarps.thread_rank () == 0)
         perBlock[blk.group_index ().x] = oddCount;
    }
  grd.sync ();
  
  //------------
  // Step-3 : reduce grid results using the first block in the grid
  if (blk.group_index ().x == 0)
    {
      oddCount = 0;
      for (int i = localID; i < grd.group_dim ().x; i += blk.size ())
          oddCount += perBlock[i];
      oddCount = reduce (tile, oddCount, cooperative_groups::plus < int >());
      blk.sync ();
      
      // save warp results to shared memory 
      if (tile.thread_rank () == 0)
          cache[tile.meta_group_rank ()] = oddCount;
      blk.sync ();

  //------------
  // Step-4 : first warp in block produces the final result
      if (tile.meta_group_rank () == 0 && localID < (blk.size () + warpSize - 1) / warpSize)
        {
          coalesced_group activeWarps = coalesced_threads ();
          oddCount = cache[localID];
          oddCount = reduce (activeWarps, oddCount, cooperative_groups::plus < int >());
          *odd = oddCount;
        }
    }
}

//------------------------------------
int main (int argc, char **argv)
{
  int N = atoi (argv[1]);       // size of array to process

  unique_ptr < int[] > h_a;     // host (h*) and
  int *d_a;                     // device (d*) pointers
  int h_odd;
  int *d_odd;
  int *d_perBlock;

  h_a = make_unique < int[] > (N);

  CUDA_CHECK_RETURN (cudaMalloc ((void **) &d_a, sizeof (int) * N));

  numberGen (N, MAXVALUE, h_a.get ());

  CUDA_CHECK_RETURN (cudaMemcpy (d_a, h_a.get (), sizeof (int) * N, cudaMemcpyHostToDevice));

  int blockSize = BLKSIZE, gridSize;
  gridSize = ceil (1.0 * N / blockSize);
  gridSize = 18;
  printf ("Grid : %i    Block : %i\n", gridSize, blockSize);

  CUDA_CHECK_RETURN (cudaMalloc ((void **) &d_odd, sizeof (int)));
  CUDA_CHECK_RETURN (cudaMemset (d_odd, 0, sizeof (int)));
  CUDA_CHECK_RETURN (cudaMalloc ((void **) &d_perBlock, sizeof (int) * gridSize));

  void *args[] = { &d_a, &N, &d_odd, &d_perBlock };
  cudaLaunchCooperativeKernel ((void *) countOdds, gridSize, blockSize, args, 0);

  CUDA_CHECK_RETURN (cudaMemcpy (&h_odd, d_odd, sizeof (int), cudaMemcpyDeviceToHost));

  printf ("#Odds : %i\n", h_odd);

  cudaFree ((void *) d_a);
  cudaFree ((void *) d_odd);
  cudaDeviceReset ();

  return 0;
}
