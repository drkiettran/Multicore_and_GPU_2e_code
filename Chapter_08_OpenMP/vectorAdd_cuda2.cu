/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : June 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP device implementation of vector operation a*(a+b)
 To build use  : nvcc vectorAdd_cuda2.cu -o vectorAdd_cuda2
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <memory>
#include <omp.h>
#include <cublas_v2.h>

using namespace std;

static const int N = 20;
static const int BLOCK_SIZE = 256;

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
  float *pa, *pb, *pc;     // host (h*) and device (d*) pointers
  int i;

  // host memory allocation
  ha = make_unique<float[]>(N);
  hb = make_unique<float[]>(N);
  hc = make_unique<float[]>(N);

  pa = ha.get();
  pb = hb.get();
  pc = hc.get();
  
  for (i = 0; i < N; i++)
    {
//       ha[i] = rand () % 10000;
//       hb[i] = rand () % 10000;
      ha[i] = i;
      hb[i] = i;
      hc[i] = ha[i]+hb[i];
    }

cublasHandle_t handle;
cublasCreate (&handle);
float alpha=2;
#pragma omp target data map(to:pa[0:N]) map(tofrom:pb[0:N]) 
// #pragma omp target data map(tofrom:pa[0:N]) map(tofrom:pb[0:N]) use_device_ptr(pa, pb)
{
#pragma omp target data use_device_ptr(pa, pb)
{
    printf("ST %f %f\n", pa[1], pb[1]);
    cublasStatus_t st = cublasSaxpy(handle, N, &alpha, pa,1, pb,1);
    printf("FINE %f %f %i\n", pa[1], pb[1], st);
}
}
printf("ST0\n");

  // correctness check
  for (i = 0; i < N; i++)
    {
      if (hc[i] != hb[i])
        printf ("Error at index %i : %f %f %f\n", i, hc[i], hb[i], ha[i]);
    }

  return 0;
}


