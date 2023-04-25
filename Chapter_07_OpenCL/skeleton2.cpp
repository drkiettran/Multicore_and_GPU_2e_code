/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2018
 License       : Released under the GNU GPL 3.0
 Description   : Bare minimum OpenCL C host program using cl_utility.h
 To build use  : (does not compile without completing the missing parts, e.g.
                 a kernel.
 ============================================================================
 */

#include <iostream>
#include <string>

#include "cl_utility.h"

using namespace std;

//============================================
void cleanUp (cl_context c, cl_command_queue q, cl_program p, cl_kernel k)
{
  if (k != 0) clReleaseKernel (k);

  if (p != 0) clReleaseProgram (p);

  if (q != 0) clReleaseCommandQueue (q);

  if (c != 0) clReleaseContext (c);

  // add other device resources that need to be released
}
//============================================
int main ()
{
  cl_int errNum;
  cl_context cont = 0;          // initialize for cleanup check
  cl_command_queue q = 0;
  cl_program pr = 0;
  cl_kernel kernel = 0;
  cl_device_id devID;
   
  setupDevice(cont, q, NULL, devID);

  // read program source and build it
  if(!setupProgramAndKernel(devID, cont, "deviceSource.cl", pr, "kernelName", kernel))
  {
      if(pr!=0)
          printf("Error: %s\n", getCompilationError(pr, devID));
      cleanUp (cont, q, pr, kernel);
      return 1;
  }

  // allocate memory on the target device and copy data from the host to it
  
  // specify the parameters that will be used for the kernel execution
  
  // setup the work item index space and group size
  size_t idxSpace[] = { N };
  size_t localWorkSize[] = { K };

  // enqueue kernel execution request
  errNum = clEnqueueNDRangeKernel (q, kernel, 1, NULL, idxSpace, localWorkSize, 0, NULL, &completeEv);

  // collect the results by copying them from the device to the host memory
  
  // release resources
  cleanUp (cont, q, pr, kernel);
  return 0;
}
