/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : July 2018
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ hello.cpp -lOpenCL -o hello
 ============================================================================
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

const int MAXNUMDEV = 10;

string kernSource = "       \
kernel void hello()   \
{                     \
   int ID = get_global_id(0);  \
   int grID = get_group_id(0);  \
   printf(\"Work item %i from group %i says hello!\\n\", ID, grID); \
}";

//============================================
void cleanUp (cl_context c, cl_command_queue q, cl_program p, cl_kernel k)
{
  if (k != 0)
    clReleaseKernel (k);

  if (p != 0)
    clReleaseProgram (p);

  if (q != 0)
    clReleaseCommandQueue (q);

  if (c != 0)
    clReleaseContext (c);
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
  if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
      cerr << "Failed to find any OpenCL platforms." << endl;
      return 1;
    }

  // Get the device IDs matching the CL_DEVICE_TYPE parameter, up to the MAXNUMDEV limit
  errNum = clGetDeviceIDs (firstPlatformId, CL_DEVICE_TYPE_ALL, MAXNUMDEV, devID, &numDev);
  if (errNum != CL_SUCCESS || numDev <= 0)
    {
      cerr << "Failed to find any OpenCL devices." << endl;
      return 2;
    }

  char devName[100];
  size_t nameLen;
  for (int i = 0; i < numDev; i++)
    {
      errNum = clGetDeviceInfo (devID[i], CL_DEVICE_NAME, 100, (void *) devName, &nameLen);
      if (errNum == CL_SUCCESS)
        cout << "Device " << i << " is " << devName << endl;
    }


  cl_context_properties prop[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties) firstPlatformId,
    0                           // termination
  };

  cont = clCreateContext (prop, numDev, devID, NULL,    // no callback function
                          NULL, // no data for callback
                          &errNum);
  if (errNum != CL_SUCCESS)
    {
      cerr << "Failed to create a context." << endl;
      cleanUp (cont, q, pr, kernel);
      return 1;
    }

  cl_queue_properties qprop[] = {
    CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
    0
  };
  q = clCreateCommandQueueWithProperties (cont, devID[0], qprop, &errNum);
  if (errNum != CL_SUCCESS)
    {
      cerr << "Failed to create a command queue" << endl;
      cleanUp (cont, q, pr, kernel);
      return 1;
    }



//     ifstream kernelFile("hello.cl", std::ios::in);
//     if (!kernelFile.is_open())
//     {
//         std::cerr << "Failed to open file for reading\n";
//         return 1;
//     }
// 
//     ostringstream oss;
//     oss << kernelFile.rdbuf();
// 
//     string srcStdStr = oss.str();
//     const char *srcStr = srcStdStr.c_str(); 
// 
//    cl_program pr = clCreateProgramWithSource (cont, 1, &srcStr, NULL, &errNum);


  const char *src = kernSource.c_str ();
  size_t len = kernSource.size ();
  pr = clCreateProgramWithSource (cont, 1, (const char **) (&src), &len, &errNum);
  if (errNum != CL_SUCCESS)
    {
      cerr << "Failed to create program." << endl;
      cleanUp (cont, q, pr, kernel);
      return 1;
    }

  errNum = clBuildProgram (pr, 1, devID, NULL, NULL, NULL);
  if (errNum != CL_SUCCESS)
    {
      cerr << "Failed to build program" << endl;
      cleanUp (cont, q, pr, kernel);
      return 1;
    }


  kernel = clCreateKernel (pr, "hello", &errNum);
  if (errNum != CL_SUCCESS || kernel == NULL)
    {
      cerr << "Failed to create kernel" << endl;
      cleanUp (cont, q, pr, kernel);
      return 1;
    }

  // work item index space and group size setup
  size_t idxSpace[] = { 12 };
  size_t localWorkSize[] = { 3 };

  cl_event completeEv;
  errNum = clEnqueueNDRangeKernel (q, kernel, 1, NULL, idxSpace, localWorkSize, 0, NULL, &completeEv);
//   errNum = clEnqueueNDRangeKernel (q, kernel, 1, NULL, idxSpace, NULL, 0, NULL, &completeEv);

  // wait for enqueued command to finish
  clWaitForEvents (1, &completeEv);

  cleanUp (cont, q, pr, kernel);
  return 0;
}
