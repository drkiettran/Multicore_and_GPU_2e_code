/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : January 2020
 License       : Released under the GNU GPL 3.0
 Description   : Image convolution in OpenCL C host program using cl_utility.h
 To build use  : icc -lOpenCL -lpthread host_threads_v1.cpp  -o host_threads_v1
 ============================================================================
 */

#include <iostream>
#include <string>
#include <time.h>
#include <thread>
#include <mutex>
#include <memory>

#include "cl_utility.h"
#include <CL/cl2.hpp>

using namespace std;

static cl_context cont = 0;     // initialize for cleanup check
static cl_command_queue q = 0;
static cl_program pr = 0;
static cl_kernel kern=0;
static mutex l;

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
void hostThr(int ID)
{
  int errNum;
  cl_event kernComplete;
  //------------------------------------------
  // specify the parameters that will be used for the kernel execution
  // setup the work item index space and group size
  size_t idxSpace[] = { 16 };
  size_t localWorkSize[] = { 16 };

  l.lock();
  // enqueue kernel execution request
  errNum = clSetKernelArg (kern, 0, sizeof (int), (void *) &ID);
  if (isError (errNum, "Failed to set kernel parameters"))
    {
      l.unlock();
      return ;
    }

  errNum = clEnqueueNDRangeKernel (q, kern, 1, NULL, idxSpace, localWorkSize, 0, NULL, &kernComplete);
  if (isError (errNum, "Failed to launch kernel"))
    {
      l.unlock();
      return ;
    }
  l.unlock();

  //------------------------------------------
  // wait for the kernel to finish
  cl_event evlist[]={kernComplete};
  clWaitForEvents(1, evlist);
}

//============================================
int main (int argc, char **argv)
{
  cl_int errNum;
  cl_device_id devID;

  if (argc < 2)
    {
      printf ("%s #num_threads\n", argv[0]);
      return 0;
    }
  int N = atoi (argv[1]);

  // context and queue setup
  setupDevice (cont, q, NULL, devID);

  //------------------------------------------
  // read program source and build it
  if (!setupProgramAndKernel (devID, cont, "hello2.cl", pr, "hello", kern))
    {
      if (pr != 0)
        printf ("Error: %s\n", getCompilationError (pr, devID));
      cleanUp ();
      return 1;
    }

  // create host threads 
  unique_ptr<thread> thr[N];
  for(int i=0;i<N;i++)
      thr[i] = make_unique<thread>(hostThr, i);
  
  
  // wait for host threads to finish
  for(int i=0;i<N;i++)
      thr[i]->join();

  // release resources
  cleanUp ();
  return 0;
}
