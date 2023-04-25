/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : June 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP device implementation of vector operation a*(a+b)
 To build use  : nvcc vectorAdd_cuda.cu -o vectorAdd_cuda
 
 g++ -fopenmp -foffload=nvptx-none -fno-stack-protector -foffload=-lm -fno-fast-math -fno-associative-math vectorAdd_cuda.cu -o vectorAdd_cuda
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <memory>
#include <omp.h>

using namespace std;

static const int N = 2000;

#define CUDA_CHECK_RETURN(value) {           \
    cudaError_t _m_cudaStat = value;         \
    if (_m_cudaStat != cudaSuccess) {        \
         fprintf(stderr, "Error %s at line %d in file %s\n",              \
                 cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);    \
         exit(1);                                                         \
       } }

int main (void)
{
  unique_ptr<float[]> ha, hb, hc;
  float *da, *db, *dc;     // host (h*) and device (d*) pointers
  int i;

  // host memory allocation
  ha = make_unique<float[]>(N);
  hb = make_unique<float[]>(N);
  hc = make_unique<float[]>(N);

  CUDA_CHECK_RETURN (cudaMalloc ((void **) &da, sizeof (float) * N));
  CUDA_CHECK_RETURN (cudaMalloc ((void **) &db, sizeof (float) * N));
  CUDA_CHECK_RETURN (cudaMalloc ((void **) &dc, sizeof (float) * N));

  for (i = 0; i < N; i++)
    {
      ha[i] = rand () % 10000;
      hb[i] = rand () % 10000;
    }
 
  // data transfer, host -> device
  CUDA_CHECK_RETURN (cudaMemcpy (da, ha.get(), sizeof (float) * N, cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN (cudaMemcpy (db, hb.get(), sizeof (float) * N, cudaMemcpyHostToDevice));

printf ("A\n");

#pragma omp target is_device_ptr(da,db,dc)
#pragma omp teams distribute parallel for
  for (int i = 0; i < N; i++)
  {
    printf("# %f %f\n",  da[i], db[i]);
    dc[i] = da[i] + db[i];
  }
printf ("B\n");

  
  // data transfer, device -> host
  CUDA_CHECK_RETURN (cudaMemcpy (hc.get(), dc, sizeof (float) * N, cudaMemcpyDeviceToHost));

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


