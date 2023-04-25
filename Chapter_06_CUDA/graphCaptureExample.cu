/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc graphCaptureExample.cu -o graphCaptureExample
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>

__global__ void A()
{
  printf ("A\n");
}
__global__ void B()
{
  printf ("B\n");
}
__global__ void C()
{
  printf ("C\n");
}
__global__ void D()
{
  printf ("D\n");
}
__global__ void E()
{
  printf ("E\n");
}
__global__ void F()
{
  printf ("F\n");
}

int main ()
{
  cudaStream_t str1, str2, execStream;
  
  cudaStreamCreate(&str1);
  cudaStreamCreate(&str2);
  cudaStreamCreate(&execStream);
  cudaGraph_t gr;
  cudaEvent_t eventB2C, eventC2D, startEvent, finishEvent;
  cudaEventCreate(&eventB2C);
  cudaEventCreate(&eventC2D);
  cudaEventCreate(&startEvent);
  cudaEventCreate(&finishEvent);
  
  // origin stream 
  cudaStreamBeginCapture(str1, cudaStreamCaptureModeGlobal);
  cudaEventRecord(startEvent, str1); 
  A<<<1,1,0,str1>>>();
  cudaStreamWaitEvent(str1, eventB2C,0);
  C<<<1,1,0,str1>>>();
  cudaEventRecord(eventC2D, str1);
  E<<<1,1,0,str1>>>();

  // second captured stream
  cudaStreamWaitEvent(str2, startEvent,0);
  B<<<1,1,0,str2>>>();
  cudaEventRecord(eventB2C, str2);
  cudaStreamWaitEvent(str2, eventC2D,0);
   D<<<1,1,0,str2>>>();
   F<<<1,1,0,str2>>>();
  cudaEventRecord(finishEvent, str2);
   
  cudaStreamWaitEvent(str1, finishEvent,0);
  cudaStreamEndCapture(str1, &gr);
  
  //***********************************************************
  // Instantiation phase
  cudaGraphExec_t instance;
  cudaGraphInstantiate (&instance, gr, NULL, NULL, 0);
  
  //***********************************************************
  // Execution phase
  cudaGraphLaunch (instance, execStream);  
  cudaStreamSynchronize(execStream);
  
  cudaEventDestroy(eventB2C);
  cudaEventDestroy(eventC2D);
  cudaStreamDestroy(str1);
  cudaStreamDestroy(str2);
  cudaDeviceReset();
  return 1;
}
