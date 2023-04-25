/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : August 2020
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc CGasync.cu -o CGasync
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
__global__ void countOdds (int *src, int N, int *odd)
{
  pipeline p;
  __shared__ int cache[2][BLKSIZE];
  int transferred[2];
  int whichBank = 0;
  int oddCount = 0;

  thread_block blk = this_thread_block ();
  grid_group grd = this_grid ();

  int myID = grd.thread_rank ();
  int localID = blk.thread_rank ();
  int totalThr = grd.size ();
  int localIdx = blockIdx.x * blockDim.x; // starting point for data copy

  // initiate the first transfer
  transferred[whichBank] = memcpy_async (blk, &(cache[whichBank][0]), BLKSIZE, src + localIdx, N - localIdx, p);
  p.commit();
  localIdx += totalThr;
  whichBank ^= 1;
  while (localIdx < N)
    {
      // initiate the next transfer
      transferred[whichBank] = memcpy_async (blk, &(cache[whichBank][0]), BLKSIZE, src + localIdx, N - localIdx, p);
      p.commit();
      localIdx += totalThr;
      whichBank ^= 1;

      // wait on the previous transfer to end
      wait (blk, p, 1);

      if (localID < transferred[whichBank]) // check if there are actual data to process
        oddCount += (cache[whichBank][localID] % 2);
    }

  // process the last batch of data
  whichBank ^= 1;
  wait (blk, p, 1);
  if (localID < transferred[whichBank])
    oddCount += (cache[whichBank][localID] % 2);

  // reduction on a per-warp basis (tile is 32 threads)
  thread_block_tile < 32 > tile = tiled_partition < 32 > (blk);
  oddCount = reduce (tile, oddCount, cooperative_groups::plus < int >());

  // first thread in the tile group adds to global counter
  if (tile.thread_rank() == 0)
    {
      atomicAdd (odd, oddCount);
    }
}

//------------------------------------
int main (int argc, char **argv)
{
  int N = atoi (argv[1]);

  unique_ptr < int[] > h_a;     // host (h*) and
  int *d_a;                     // device (d*) pointers
  int h_odd;
  int *d_odd;

  h_a = make_unique < int[] > (N);

  CUDA_CHECK_RETURN (cudaMalloc ((void **) &d_a, sizeof (int) * N));

  numberGen (N, MAXVALUE, h_a.get ());

  CUDA_CHECK_RETURN (cudaMemcpy (d_a, h_a.get (), sizeof (int) * N, cudaMemcpyHostToDevice));

  int blockSize = BLKSIZE, gridSize;
  gridSize = ceil (1.0 * N / blockSize);
  gridSize = 1;
  printf ("Grid : %i    Block : %i\n", gridSize, blockSize);

  CUDA_CHECK_RETURN (cudaMalloc ((void **) &d_odd, sizeof (int)));
  CUDA_CHECK_RETURN (cudaMemset (d_odd, 0, sizeof (int)));

  countOdds <<< gridSize, blockSize >>> (d_a, N, d_odd);

  CUDA_CHECK_RETURN (cudaMemcpy (&h_odd, d_odd, sizeof (int), cudaMemcpyDeviceToHost));

  printf ("#Odds : %i\n", h_odd);

  cudaFree ((void *) d_a);
  cudaFree ((void *) d_odd);
  cudaDeviceReset ();

  return 0;
}
