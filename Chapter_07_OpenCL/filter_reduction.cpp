/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : January 2020
 License       : Released under the GNU GPL 3.0
 Description   : Image convolution in OpenCL C host program using cl_utility.h
 To build use  : icc -lOpenCL filter_reduction.cpp -o filter_reduction
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
static cl_kernel kern1 = 0;
static cl_kernel kern2 = 0;
static int *data = 0;
static int *filteredData = 0;
static int *groupCounts = 0;

//============================================
void cleanUp ()
{
  if (kern1 != 0)
    clReleaseKernel (kern1);

  if (kern2 != 0)
    clReleaseKernel (kern2);

  if (pr != 0)
    clReleaseProgram (pr);

  if (q != 0)
    clReleaseCommandQueue (q);

  if (cont != 0)
    {
      if (data)
        clSVMFree (cont, data);

      if (filteredData)
        clSVMFree (cont, filteredData);

      if (groupCounts)
        clSVMFree (cont, groupCounts);

      clReleaseContext (cont);
    }
}

//============================================
int main (int argc, char **argv)
{
  cl_int errNum;
  cl_device_id devID;
  cl_event countKernComplete, moveKernComplete;

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
  if (!setupProgramAndKernel (devID, cont, "filter_reduction.cl", pr, "countOdds", kern1))
    {
      if (pr != 0)
        printf ("Error: %s\n", getCompilationError (pr, devID));
      cleanUp ();
      return 1;
    }

  // second kernel build from the same source    
  kern2 = clCreateKernel (pr, "moveOdds", &errNum);
  if (isError (errNum, "Failed to create kernel2."))
    {
      cleanUp ();
      return 1;
    }

  //------------------------------------------
  // SVM memory allocation
  int *data = (int *) clSVMAlloc (cont, CL_MEM_READ_WRITE, N * sizeof (int), 0);
  if (data == NULL)
    {
      cleanUp ();
      return 1;
    }
  for (int i = 0; i < N; i++)
    data[i] = i;

  int *filteredData = (int *) clSVMAlloc (cont, CL_MEM_READ_WRITE, N * sizeof (int), 0);
  if (filteredData == NULL)
    {
      cleanUp ();
      return 1;
    }

  int numWorkGroups = (N + 255) / 256;
  int *groupCounts = (int *) clSVMAlloc (cont, CL_MEM_READ_WRITE, (numWorkGroups + 1) * sizeof (int), 0);
  if (groupCounts == NULL)
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
  errNum = clSetKernelArgSVMPointer (kern1, 0, data);
  errNum |= clSetKernelArg (kern1, 1, sizeof (int), (void *) &N);
  errNum |= clSetKernelArgSVMPointer (kern1, 2, groupCounts);
  if (isError (errNum, "Failed to set kernel1 parameters"))
    {
      cleanUp ();
      return 1;
    }

  errNum = clEnqueueNDRangeKernel (q, kern1, 1, NULL, idxSpace, localWorkSize, 0, NULL, &countKernComplete);
  if (isError (errNum, "Failed to launch kernel1"))
    {
      cleanUp ();
      return 1;
    }

  //------------------------------------------
  // wait for the kernel to finish
  cl_event evlist[] = { countKernComplete };
  errNum = clWaitForEvents (1, evlist);
  if (isError (errNum, "Failed to wait for event"))
    {
      cleanUp ();
      return 1;
    }

  //------------------------------------------
  // calculate the prefix-sum for the group offsets
  groupCounts[0] = 0;
  for (int i = 0; i < numWorkGroups; i++)
    groupCounts[i + 1] += groupCounts[i];

  //------------------------------------------ 
  // set parameters and call the second kernel  
  errNum = clSetKernelArgSVMPointer (kern2, 0, data);
  errNum |= clSetKernelArg (kern2, 1, sizeof (int), (void *) &N);
  errNum |= clSetKernelArgSVMPointer (kern2, 2, filteredData);
  errNum |= clSetKernelArgSVMPointer (kern2, 3, groupCounts);
  if (isError (errNum, "Failed to set kernel2 parameters"))
    {
      cleanUp ();
      return 1;
    }

  errNum = clEnqueueNDRangeKernel (q, kern2, 1, NULL, idxSpace, localWorkSize, 0, NULL, &moveKernComplete);
  if (isError (errNum, "Failed to launch kernel"))
    {
      cleanUp ();
      return 1;
    }

  //------------------------------------------
  // wait for the kernel to finish
  cl_event evlist2[] = { moveKernComplete };
  errNum = clWaitForEvents (1, evlist2);
  if (isError (errNum, "Failed to wait for event"))
    {
      cleanUp ();
      return 1;
    }

  for (int i = 0; i < groupCounts[numWorkGroups]; i++)
    printf ("%i ", filteredData[i]);
  printf ("\n");

  // release resources
  cleanUp ();
  return 0;
}
