/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.1
 Last modified : November 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc warpFix_dataRearr.cu -o warpFix_dataRearr
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <cuda.h>

using namespace std;

#define MAXVALUE 10000

//------------------------------------
void numberGen (int N, int max, int *store)
{
  int i;
  srand (time (0));
  for (i = 0; i < N; i++)
    store[i] = rand () % max;
}

__device__ int doSmt (int x) { return x; }

__device__ int doSmtElse (int x) { return x; }

//------------------------------------

__global__ void foo (int *d, int N, int *r)
{
  int myID = blockIdx.x * blockDim.x + threadIdx.x;
  if (myID < N)
    {
      if (d[myID] % 2 == 0)
        r[myID] = doSmt (d[myID]);
      else
        r[myID] = doSmtElse (d[myID]);
    }
}

//------------------------------------

int main (int argc, char **argv)
{
  int N = atoi (argv[1]);

  unique_ptr<int[]> ha, hres, hresOrdered;     // host (h*) and device (d*) pointers
  int *da, *dres;     // host (h*) and device (d*) pointers

  ha = make_unique<int[]>(N);
  hres = make_unique<int[]>(N);
  hresOrdered = make_unique<int[]>(N);

  cudaMalloc ((void **) &da, sizeof (int) * N);
  cudaMalloc ((void **) &dres, sizeof (int) * N);

  numberGen (N, MAXVALUE, ha.get());

/*
  for(int i=0;i<N ;i++)
   printf("%i ", ha[i]);
  printf("\n");*/

  // rearrange data
  int evenIdx = 0, oddIdx = N - 1;
  int *origPos = new int[N];
  for (int i = 0; i < N; i++)
    origPos[i] = i;

  for (int i = 0; i < N && evenIdx < oddIdx; i++)
    {
      if (ha[i] % 2 != 0)
        {
          int tmp = ha[i];
          ha[i] = ha[oddIdx];
          ha[oddIdx] = tmp;
          tmp = origPos[i];
          origPos[i] = origPos[oddIdx];
          origPos[oddIdx] = tmp;
          i--;
          oddIdx--;
        }
      else
        evenIdx++;
    }

//   for(int i=0;i<N ;i++)
//    printf("%i ", origPos[i]);
//   printf("\n");
//   for(int i=0;i<N ;i++)
//    printf("%i ", ha[i]);
//   printf("\n");


  cudaMemcpy (da, ha.get(), sizeof (int) * N, cudaMemcpyHostToDevice);
  cudaMemset (dres, 0, sizeof (int));

  int blockSize = 256, gridSize;
  gridSize = ceil (1.0 * N / blockSize);
  foo <<< gridSize, blockSize >>> (da, N, dres);

  cudaMemcpy (hres.get(), dres, N * sizeof (int), cudaMemcpyDeviceToHost);

  // restore original placement  
  for (int i = 0; i < N; i++)
      hresOrdered[origPos[i]] = hres[i];

//   for(int i=0;i<N ;i++)
//    printf("%i ", hresOrdered[i]);
//   printf("\n");

  cudaFree ((void *) da);
  cudaFree ((void *) dres);
  cudaDeviceReset ();

  return 0;
}
