/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : August 2020
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc sharedMemoryTest.cu -o sharedMemoryTest
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>

__global__ void testShared ()
{
  __shared__ unsigned char count[256];  

  int myID = threadIdx.x; 
  count[myID]=0;
  __syncthreads();
  for(int i=0;i<100;i++)
      count[myID]++; // four threads write to different parts of the same 32-bit word
  
  __syncthreads();
  if(myID==0)
  {
    for(int i=0;i<256;i++)
        if(count[i]!=100)
           printf ("Error at %i (%i)\n", i, count[i]);
  }
}

//--------------------------------
__global__ void testShared2 ()
{
  __shared__ int count[64];  

  int myID = threadIdx.x; 
  if(myID<64) count[myID]=0;
  __syncthreads();
  for(int i=0;i<100;i++)
      count[myID / 4]++; // four threads write to the same 32-bit word

  __syncthreads();
  if(myID==0)
  {
    for(int i=0;i<64;i++)
        if(count[i]!=400)
           printf ("Error2 at %i (%i)\n", i, count[i]);
  }
}

//--------------------------------
int main ()
{
  testShared <<< 1, 256 >>> ();
  testShared2 <<< 1, 256 >>> ();
  cudaDeviceSynchronize ();
  return 0;
}
