/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : November 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc odd__maxActive.cu -o odd_maxActive
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
  int totalThr = blockDim.x * gridDim.x;
  int localID = threadIdx.x;
  count[localID] = 0;
  for(int i=myID; i < N; i+= totalThr)
     count[localID] += (d[i] % 2);
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

  cudaDeviceProp pr;
  cudaGetDeviceProperties (&pr, 0);     // replace 0 with appropriate ID in case of a multi-GPU system
  int SM = pr.multiProcessorCount;
  
  int blockSize = 256;
  int blockPerSM, gridSize;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockPerSM, (void *) countOdds, blockSize, sharedSize(blockSize));
   
  gridSize = min( (int)ceil (1.0 * N / blockSize), blockPerSM * SM);

  printf ("Grid : %i    Block : %i  Suggested: %i\n", gridSize, blockSize, blockPerSM * SM);
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
