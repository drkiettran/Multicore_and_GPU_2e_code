/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc odd_CGfilter.cu -o odd_CGfilter
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
    store[i] = i;//rand () % max;
}

//------------------------------------
__global__ void countOdds (int *d, int N, int *oddBlockCount)
{
  __shared__ int count;

  int myID = blockIdx.x * blockDim.x + threadIdx.x;
  int localID = threadIdx.x;
  bool isOdd=false;
  if(localID==0)
    count=0;
  __syncthreads ();
    
  if (myID < N)
    isOdd = d[myID] % 2;

  if(isOdd)
   {
     coalesced_group active = coalesced_threads();
     if(active.thread_rank()==0)
       atomicAdd (&count, active.size());
   }

  __syncthreads ();

  if (localID == 0)
     oddBlockCount[blockIdx.x] = count;        
}
//------------------------------------
__global__ void moveOdds (int *src, int N, int *dest, int *oddBlockCount)
{
  int myID = blockIdx.x * blockDim.x + threadIdx.x;
  bool isOdd=false;
  if (myID < N)
    isOdd = src[myID] % 2;
//   __syncthreads ();

  if(isOdd)
   {
     coalesced_group active = coalesced_threads();
     int offset;
     if(active.thread_rank() == 0)
       offset = atomicAdd(oddBlockCount + blockIdx.x, active.size());
     offset = active.shfl(offset,0);  
     dest[offset + active.thread_rank() ] = src[myID];
   }
}

//------------------------------------
int main (int argc, char **argv)
{
  int N = atoi (argv[1]);

  unique_ptr<int[]> h_a; // host (h*) and
  unique_ptr<int[]> h_odd; 
  unique_ptr<int[]> h_blockCounts;
  int *d_a;              // device (d*) pointers
  int *d_odd;
  int *d_blockCounts;
  
  h_a = make_unique<int[]>(N);

  cudaMalloc ((void **) &d_a, sizeof (int) * N);

  numberGen (N, MAXVALUE, h_a.get());

//   for (int i = 0; i < N; i++)
//     printf("%i ", h_a[i]);
//   printf ("\n");
  cudaMemcpy (d_a, h_a.get(), sizeof (int) * N, cudaMemcpyHostToDevice);

  int blockSize, gridSize;
  cudaOccupancyMaxPotentialBlockSize (&gridSize, &blockSize, (void *) countOdds, sizeof(int), N);

  gridSize = ceil (1.0 * N / blockSize);
  printf ("Grid : %i    Block : %i\n", gridSize, blockSize);

  // allocate as many counters as blocks
  h_blockCounts = make_unique<int[]>(gridSize);
  cudaMalloc ((void **) &d_blockCounts, sizeof (int) * gridSize);
  
  // first count what each block is supposed to handle
  countOdds <<< gridSize, blockSize >>> (d_a, N, d_blockCounts);

  cudaMemcpy (h_blockCounts.get(), d_blockCounts, sizeof (int)*gridSize, cudaMemcpyDeviceToHost);

  // exclusive scan or pre-scan calculation
  int toAdd=0;
  for(int i=0;i<gridSize;i++)
  {
     int tmp = h_blockCounts[i];
     h_blockCounts[i] = toAdd;     
     toAdd += tmp;
  }
  
  // offsets for each block copied back to the device
  cudaMemcpy (d_blockCounts, h_blockCounts.get(), sizeof (int)*gridSize, cudaMemcpyHostToDevice);

  // allocate memory for the result-holding array
  h_odd = make_unique<int[]>(toAdd);
  cudaMalloc ((void **) &d_odd, sizeof (int) * toAdd);
  
  moveOdds <<< gridSize, blockSize>>> (d_a, N, d_odd, d_blockCounts);
  
  cudaMemcpy (h_odd.get(), d_odd, sizeof (int)*toAdd, cudaMemcpyDeviceToHost);

  for (int i = 0; i < toAdd; i++)
    printf("%i ", h_odd[i]);
      
  printf ("\n");

  cudaFree ((void *) d_a);
  cudaFree ((void *) d_odd);
  cudaFree ((void *) d_blockCounts);
  cudaDeviceReset ();

  return 0;
}
