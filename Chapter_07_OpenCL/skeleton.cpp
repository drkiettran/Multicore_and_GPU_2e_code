/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2018
 License       : Released under the GNU GPL 3.0
 Description   : Bare minimum OpenCL C host program
 To build use  : (does not compile without completing the missing parts, e.g.
                 a kernel.
 ============================================================================
 */

#include <iostream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

const int MAXNUMDEV = 10;

string kernSource = ""; // static OpenCL source, or it can be read from a file

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
  cl_uint numPlatforms;
  cl_platform_id firstPlatformId;
  cl_device_id devID[MAXNUMDEV];
  cl_uint numDev;
  cl_context cont = 0;          // initialize for cleanup check
  cl_command_queue q = 0;
  cl_program pr = 0;
  cl_kernel kernel = 0;

  // Get a reference to an object representing a platform 
  errNum = clGetPlatformIDs (1, &firstPlatformId, &numPlatforms);

  // Get the device IDs matching the CL_DEVICE_TYPE parameter, up to the MAXNUMDEV limit
  errNum = clGetDeviceIDs (firstPlatformId, CL_DEVICE_TYPE_ALL, MAXNUMDEV, devID, &numDev);

  cl_context_properties prop[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties) firstPlatformId,
    0                           // termination
  };
  
  // create a context for the devices detected
  cont = clCreateContext (prop, numDev, devID, NULL, NULL, &errNum);

  // create a command queue
  q = clCreateCommandQueueWithProperties (cont, devID[0], NULL, &errNum);

  // create a program object with supplied OpenCL source
  const char *src = kernSource.c_str ();
  size_t len = kernSource.size ();
  pr = clCreateProgramWithSource (cont, 1, (const char **) (&src), &len, &errNum);

  // compile the program
  errNum = clBuildProgram (pr, 1, devID, NULL, NULL, NULL);

  // create a kernel object
  kernel = clCreateKernel (pr, "hello", &errNum);

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
