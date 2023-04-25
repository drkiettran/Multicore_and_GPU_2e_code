/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : November 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc vectorAdd.cu -o vectorAdd
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <memory>

using namespace std;

static const int BLOCK_SIZE = 256;
static const int N = 2000;

#define CUDA_CHECK_RETURN(value) {           \
    cudaError_t _m_cudaStat = value;         \
    if (_m_cudaStat != cudaSuccess) {        \
         fprintf(stderr, "Error %s at line %d in file %s\n",              \
                 cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);    \
         exit(1);                                                         \
       } }

// kernel that calculates only one element of the result
__global__ void vadd (int *a, int *b, int *c, int N)
{
  int myID = blockIdx.x * blockDim.x + threadIdx.x;
  if (myID < N)
    c[myID] = a[myID] + b[myID];
}

int main (void)
{
  unique_ptr<int[]> ha, hb, hc;
  int *da, *db, *dc;     // host (h*) and device (d*) pointers
  int i;

  // host memory allocation
  ha = make_unique<int[]>(N);
  hb = make_unique<int[]>(N);
  hc = make_unique<int[]>(N);

  CUDA_CHECK_RETURN (cudaMalloc ((void **) &da, sizeof (int) * N));
  CUDA_CHECK_RETURN (cudaMalloc ((void **) &db, sizeof (int) * N));
  CUDA_CHECK_RETURN (cudaMalloc ((void **) &dc, sizeof (int) * N));

  for (i = 0; i < N; i++)
    {
      ha[i] = rand () % 10000;
      hb[i] = rand () % 10000;
    }
 
  // data transfer, host -> device
  CUDA_CHECK_RETURN (cudaMemcpy (da, ha.get(), sizeof (int) * N, cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN (cudaMemcpy (db, hb.get(), sizeof (int) * N, cudaMemcpyHostToDevice));

  int grid = ceil (N * 1.0 / BLOCK_SIZE);
  vadd <<< grid, BLOCK_SIZE >>> (da, db, dc, N);

  CUDA_CHECK_RETURN (cudaDeviceSynchronize ());
  // Wait for the GPU launched work to complete
  CUDA_CHECK_RETURN (cudaGetLastError ());
  
  // data transfer, device -> host
  CUDA_CHECK_RETURN (cudaMemcpy (hc.get(), dc, sizeof (int) * N, cudaMemcpyDeviceToHost));

  // correctness check
  for (i = 0; i < N; i++)
    {
      if (hc[i] != ha[i] + hb[i])
        printf ("Error at index %i : %i VS %i\n", i, hc[i], ha[i] + hb[i]);
    }

  CUDA_CHECK_RETURN (cudaFree ((void *) da));
  CUDA_CHECK_RETURN (cudaFree ((void *) db));
  CUDA_CHECK_RETURN (cudaFree ((void *) dc));
  CUDA_CHECK_RETURN (cudaDeviceReset ());

  return 0;
}
