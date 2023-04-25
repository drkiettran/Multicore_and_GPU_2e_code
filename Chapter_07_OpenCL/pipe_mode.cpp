/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : January 2020
 License       : Released under the GNU GPL 3.0
 Description   : Image convolution in OpenCL C host program using cl_utility.h
 To build use  : icc -lOpenCL pipe_mode.cpp -o pipe_mode
 ============================================================================
 */

#include <iostream>
#include <string>
#include <time.h>

#include "cl_utility.h"
#include <CL/cl2.hpp>

using namespace std;

static cl_context cont = 0;     // initialize for cleanup check
static cl_command_queue q = 0;
static cl_program pr = 0;
static cl_kernel countKernel = 0, reduceKernel;
static cl_mem d_mode = 0, d_countsPipe = 0, d_data = 0;

//============================================
void cleanUp ()
{
  if (d_mode != 0)
    clReleaseMemObject (d_mode);

  if (d_countsPipe != 0)
    clReleaseMemObject (d_countsPipe);

  if (d_data != 0)
    clReleaseMemObject (d_data);

  if (countKernel != 0)
    clReleaseKernel (countKernel);

  if (countKernel != 0)
    clReleaseKernel (countKernel);

  if (pr != 0)
    clReleaseProgram (pr);

  if (q != 0)
    clReleaseCommandQueue (q);

  if (cont != 0)
    clReleaseContext (cont);
}

//============================================
int main (int argc, char **argv)
{
  cl_int errNum;
  cl_device_id devID;
  cl_event kernComplete;

  if (argc < 4)
    {
      printf ("%s #num_data lowerValue higherValue\n", argv[0]);
      return 0;
    }
  int N = atoi (argv[1]);
  int st = atoi (argv[2]);
  int end = atoi (argv[3]);

  // context and queue setup
  setupDevice (cont, q, NULL, devID);

  //------------------------------------------
  // read program source and build it
  if (!setupProgramAndKernel (devID, cont, "pipe_mode.cl", pr, "freqCount", countKernel))
    {
      if (pr != 0)
        printf ("Error: %s\n", getCompilationError (pr, devID));
      cleanUp ();
      return 1;
    }

  // second kernel build from the same source    
  reduceKernel = clCreateKernel (pr, "modeFind", &errNum);
  if (isError (errNum, "Failed to create reduce kernel."))
    {
      cleanUp ();
      return 1;
    }

  //------------------------------------------
  // host-side memory allocation & I/O
  int *h_data = new int[N];
  srand (clock ());
  for (int i = 0; i < N; i++)
    h_data[i] = st + rand () % (end - st + 1);

  //------------------------------------------
  // allocate memory on the target device and copy data from the host to it
  int totalWorkItems = (N / 10 + 1);
  int numWorkGroups = (totalWorkItems+255)/256;
  int pipeSize = numWorkGroups * (end - st + 1); 
  d_countsPipe = clCreatePipe (cont, 0, 2 * sizeof (int), pipeSize, NULL, &errNum);
  if (isError (errNum, "Failed to create pipe"))
    {
      cleanUp ();
      return 1;
    }

  // Device input data
  d_data = clCreateBuffer (cont, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N * sizeof (int), h_data, &errNum);
  if (isError (errNum, "Failed to create device input data buffer"))
    {
      cleanUp ();
      return 1;
    }

  // Device mode result
  d_mode = clCreateBuffer (cont, CL_MEM_READ_WRITE, 2 * sizeof (int), NULL, &errNum);
  if (isError (errNum, "Failed to create device mode storage"))
    {
      cleanUp ();
      return 1;
    }

  //------------------------------------------
  // specify the parameters that will be used for the kernel execution
  // setup the work item index space and group size
  size_t idxSpace_k1[] = { totalWorkItems };
  size_t localWorkSize_k1[] = { 256 };

  size_t idxSpace_k2[] = { 256 };
  size_t localWorkSize_k2[] = { 256 };

  // enqueue kernel execution request
  errNum = clSetKernelArg (countKernel, 0, sizeof (cl_mem), (void *) &d_data);
  errNum |= clSetKernelArg (countKernel, 1, sizeof (int), (void *) &N);
  errNum |= clSetKernelArg (countKernel, 2, sizeof (cl_mem), (void *) &(d_countsPipe));
  errNum |= clSetKernelArg (countKernel, 3, sizeof (int), (void *) &st);
  errNum |= clSetKernelArg (countKernel, 4, sizeof (int), (void *) &end);
  errNum |= clSetKernelArg (countKernel, 5, 256 * sizeof (int), NULL);
  if (isError (errNum, "Failed to set kernel parameters for freqCount"))
    {
      cleanUp ();
      return 1;
    }

  errNum = clSetKernelArg (reduceKernel, 0, sizeof (cl_mem), (void *) &d_countsPipe);
  errNum |= clSetKernelArg (reduceKernel, 1, sizeof (cl_mem), (void *) &d_mode);
  errNum |= clSetKernelArg (reduceKernel, 2, sizeof (int), (void *) &N);
  errNum |= clSetKernelArg (reduceKernel, 3, sizeof (int), (void *) &st);
  errNum |= clSetKernelArg (reduceKernel, 4, sizeof (int), (void *) &end);
  errNum |= clSetKernelArg (reduceKernel, 5, 256 * sizeof (int), NULL);
  if (isError (errNum, "Failed to set kernel parameters for modeFind"))
    {
      cleanUp ();
      return 1;
    }

  errNum = clEnqueueNDRangeKernel (q, countKernel, 1, NULL, idxSpace_k1, localWorkSize_k1, 0, NULL, NULL);
  if (isError (errNum, "Failed to launch freqCount"))
    {
      cleanUp ();
      return 1;
    }

  errNum = clEnqueueNDRangeKernel (q, reduceKernel, 1, NULL, idxSpace_k2, localWorkSize_k2, 0, NULL, &kernComplete);
  if (isError (errNum, "Failed to launch modeFind"))
    {
      cleanUp ();
      return 1;
    }

  //------------------------------------------
  // collect the results by copying them from the device to the host memory
  int h_res[2];
  errNum = clEnqueueReadBuffer (q, d_mode, CL_TRUE, 0, 2 * sizeof (int), h_res, 0, NULL, NULL);
  if (isError (errNum, "Failed to get the results"))
    {
      cleanUp ();
      return 1;
    }

  printf ("Mode %i with %i occur.\n", h_res[0], h_res[1]);
  // release resources
  cleanUp ();
  delete[]h_data;
  return 0;
}
