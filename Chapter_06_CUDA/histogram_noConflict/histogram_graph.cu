/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : November 2019
 License       : Released under the GNU GPL 3.0
 Description   : Maximum number of bins are used
                 warpSize is assumed to be fixed to 32
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <memory>
#include <cuda.h>
#include "../common/pgm.h"

using namespace std;

const int BINS = 256;
const int BINS4ALL = BINS * 32;

//*****************************************************************
void CPU_histogram (const unsigned char *__restrict__ in, int N, int *__restrict__ h, int bins)
{
  int i;
  // initialize histogram counts
  for (i = 0; i < bins; i++)
    h[i] = 0;

  // accummulate counts
  for (i = 0; i < N; i++)
    h[in[i]]++;
}

//*****************************************************************
__global__ void GPU_histogram_atomic (int *in, int N, int *h)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  int locID = threadIdx.x;
  int GRIDSIZE = gridDim.x * blockDim.x;
  __shared__ int localH[BINS4ALL];
  int bankID = locID % warpSize;
  int i;

  // initialize the local, shared-memory bins
  for (i = locID; i < BINS4ALL; i += blockDim.x)
    localH[i] = 0;

  // wait for all warps to complete the previous step
  __syncthreads ();

  //start processing the image data
  int *mySharedBank = localH + bankID;
  if (blockDim.x > warpSize)    // if the blocksize exceeds the warpSize, it is possible multiple warps run at the same time
    for (i = gloID; i < N; i += GRIDSIZE)
      {

        int temp = in[i];
        int v = temp & 0xFF;
        int v2 = (temp >> 8) & 0xFF;
        int v3 = (temp >> 16) & 0xFF;
        int v4 = (temp >> 24) & 0xFF;
        atomicAdd (mySharedBank + (v << 5), 1);
        atomicAdd (mySharedBank + (v2 << 5), 1);
        atomicAdd (mySharedBank + (v3 << 5), 1);
        atomicAdd (mySharedBank + (v4 << 5), 1);
      }
  else
    for (i = gloID; i < N; i += GRIDSIZE)
      {

        int temp = in[i];
        int v = temp & 0xFF;
        int v2 = (temp >> 8) & 0xFF;
        int v3 = (temp >> 16) & 0xFF;
        int v4 = (temp >> 24) & 0xFF;
        mySharedBank[v << 5]++; // Optimized version of localH[bankID + v * warpSize]++
        mySharedBank[v2 << 5]++;
        mySharedBank[v3 << 5]++;
        mySharedBank[v4 << 5]++;
      }

  // wait for all warps to complete the local calculations, before updating the global counts
  __syncthreads ();

  // use atomic operations to add the local findings to the global memory bins 
  for (i = locID; i < BINS4ALL; i += blockDim.x)
    atomicAdd (h + (i >> 5), localH[i]);        // Optimized version of atomicAdd (h + (i/warpSize), localH[i]);
}

//*****************************************************************
int main (int argc, char **argv)
{
  PGMImage inImg (argv[1]);

  unique_ptr < int[] > h_hist, cpu_hist;
  int *d_in, *h_in;
  int *d_hist;
  int i, N, bins;

  h_in = (int *) inImg.pixels;
  N = ceil ((inImg.x_dim * inImg.y_dim) / 4.0);

  bins = inImg.num_colors + 1;
  h_hist = make_unique < int[] > (bins);
  cpu_hist = make_unique < int[] > (bins);

  CPU_histogram (inImg.pixels, inImg.x_dim * inImg.y_dim, cpu_hist.get (), bins);

  cudaMalloc ((void **) &d_in, sizeof (int) * N);
  cudaMalloc ((void **) &d_hist, sizeof (int) * bins);

  cudaDeviceProp pr;
  cudaGetDeviceProperties (&pr, 0);     // replace 0 with appropriate ID in case of a multi-GPU system
  int SM = pr.multiProcessorCount;
  int blockSize = 256;
  int gridSize = min (SM, (int) ceil (1.0 * N / blockSize));

  printf ("Grid : %i\n", gridSize);

  cudaStream_t str;
  cudaStreamCreate (&str);

  //***********************************************************
  // Creation phase
  cudaGraph_t gr;
  cudaGraphCreate (&gr, 0);

  //------------------------
  // Kernel node:
  cudaGraphNode_t kern;
  cudaKernelNodeParams kernParms;
  void *paramList[3] = { &d_in, &N, &d_hist };

  kernParms.func = (void *) GPU_histogram_atomic;
  kernParms.gridDim = gridSize;
  kernParms.blockDim = blockSize;
  kernParms.sharedMemBytes = 0;
  kernParms.kernelParams = paramList;
  kernParms.extra = NULL;

  cudaGraphAddKernelNode (&kern, gr, NULL, 0, &kernParms);

  //------------------------
  // Host-to-device transfer node:
  cudaGraphNode_t h2d;
  cudaMemcpy3DParms h2dParms;

  h2dParms.srcArray = NULL;
  h2dParms.srcPtr = make_cudaPitchedPtr ((void *) h_in, sizeof (int) * N, N, 1);
  h2dParms.srcPos = make_cudaPos (0, 0, 0);
  h2dParms.dstArray = NULL;
  h2dParms.dstPtr = make_cudaPitchedPtr ((void *) d_in, sizeof (int) * N, N, 1);
  h2dParms.dstPos = make_cudaPos (0, 0, 0);
  h2dParms.extent = make_cudaExtent (sizeof (int) * N, 1, 1);
  h2dParms.kind = cudaMemcpyHostToDevice;

  cudaGraphAddMemcpyNode (&h2d, gr, NULL, 0, &h2dParms);

  //------------------------
  // Device memory set node:
  cudaGraphNode_t dset;
  cudaMemsetParams dsetParms;

  dsetParms.dst = d_hist;
  //dsetParms.pitch = 
  dsetParms.value = 0;
  dsetParms.elementSize = 4;
  dsetParms.width = bins;
  dsetParms.height = 1;

  cudaGraphAddMemsetNode (&dset, gr, NULL, 0, &dsetParms);

  //------------------------
  // Device-to-host transfer node:
  cudaGraphNode_t d2h;
  cudaMemcpy3DParms d2hParms;

  d2hParms.srcArray = NULL;
  d2hParms.srcPtr = make_cudaPitchedPtr ((void *) d_hist, sizeof (int) * bins, bins, 1);
  d2hParms.srcPos = make_cudaPos (0, 0, 0);
  d2hParms.dstArray = NULL;
  d2hParms.dstPtr = make_cudaPitchedPtr ((void *) h_hist.get (), sizeof (int) * bins, bins, 1);
  d2hParms.dstPos = make_cudaPos (0, 0, 0);
  d2hParms.extent = make_cudaExtent (sizeof (int) * bins, 1, 1);
  d2hParms.kind = cudaMemcpyDeviceToHost;

  cudaGraphAddMemcpyNode (&d2h, gr, NULL, 0, &d2hParms);
  //------------------------
  // Dependencies 
  cudaGraphNode_t from[] = { h2d, dset, kern };
  cudaGraphNode_t to[] = { kern, kern, d2h };

  cudaGraphAddDependencies (gr, from, to, 3);

  //***********************************************************
  // Instantiation phase
  cudaGraphNode_t errorNode;
  cudaGraphExec_t instance;
  const int LOGSIZE = 100;
  char log[LOGSIZE];
  cudaGraphInstantiate (&instance, gr, &errorNode, log, LOGSIZE);

  //***********************************************************
  // Execution phase
  cudaGraphLaunch (instance, str);

  cudaStreamSynchronize (str);

  for (i = 0; i < BINS; i++)
    if (cpu_hist[i] != h_hist[i])
      printf ("Calculation mismatch (static) at : %i\n", i);

  // clean-up allocated objects and reset device
  cudaStreamDestroy (str);
  cudaGraphDestroy (gr);
  cudaFree ((void *) d_in);
  cudaFree ((void *) d_hist);
  cudaDeviceReset ();

  return 0;
}
