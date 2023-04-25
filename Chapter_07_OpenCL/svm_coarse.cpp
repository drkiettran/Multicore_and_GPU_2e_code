/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : January 2020
 License       : Released under the GNU GPL 3.0
 Description   : Image convolution in OpenCL C host program using cl_utility.h
 To build use  : icc -lOpenCL svm_coarse.cpp -o svm_coarse
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
static cl_kernel kern=0;

//============================================
void cleanUp ()
{
  if (kern != 0)
    clReleaseKernel (kern);

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

  if (argc < 2)
    {
      printf ("%s #num_data\n", argv[0]);
      return 0;
    }
  int N = atoi (argv[1]);

  // context and queue setup
  setupDevice (cont, q, NULL, devID);

  //------------------------------------------
  // read program source and build it
  if (!setupProgramAndKernel (devID, cont, "svm_coarse.cl", pr, "initKernel", kern))
    {
      if (pr != 0)
        printf ("Error: %s\n", getCompilationError (pr, devID));
      cleanUp ();
      return 1;
    }

  //------------------------------------------
  // host-side memory allocation & I/O
  int *data = (int *)clSVMAlloc(cont, CL_MEM_READ_WRITE, N*sizeof(int),0);
  if (data == NULL)
    {
      cleanUp ();
      return 1;
    }

  //------------------------------------------
  // specify the parameters that will be used for the kernel execution
  // setup the work item index space and group size
  size_t idxSpace[] = { N };
  size_t localWorkSize[] = { 256 };

  // enqueue kernel execution request
  errNum = clSetKernelArgSVMPointer(kern, 0, data);
  errNum |= clSetKernelArg (kern, 1, sizeof (int), (void *) &N);
  if (isError (errNum, "Failed to set kernel parameters"))
    {
      cleanUp ();
      return 1;
    }


  errNum = clEnqueueNDRangeKernel (q, kern, 1, NULL, idxSpace, localWorkSize, 0, NULL, &kernComplete);
  if (isError (errNum, "Failed to launch kernel"))
    {
      cleanUp ();
      return 1;
    }

  //------------------------------------------
  // wait for the kernel to finish
  cl_event evlist[]={kernComplete};
  errNum = clWaitForEvents(1, evlist);
  if (isError (errNum, "Failed to wait for event"))
    {
      cleanUp ();
      return 1;
    }
   
  for(int i=0;i<N;i++)
      printf("%i ",data[i]);
  printf("\n");
  
  // release resources
  clSVMFree(cont, data);
  cleanUp ();
  return 0;
}
