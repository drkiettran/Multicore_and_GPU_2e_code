/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : August 2018
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ vectorAdd.cpp -lOpenCL -o vectorAdd
 ============================================================================
 */

#include <iostream>
#include <string>
#include <assert.h>

#include "cl_utility.h"

using namespace std;


const int VECSIZE = 100;
const size_t DATASIZE = VECSIZE*sizeof(int);

//============================================
void cleanUp (cl_context c, cl_command_queue q, cl_program p, cl_kernel k, cl_mem d_A, cl_mem d_B, cl_mem d_C)
{
  if(d_A != 0)
    clReleaseMemObject(d_A);

  if(d_B != 0)
    clReleaseMemObject(d_B);

  if(d_C != 0)
    clReleaseMemObject(d_C);   

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
  cl_context cont = 0;          // initialize for cleanup check
  cl_command_queue q = 0;
  cl_program pr = 0;
  cl_kernel kernel = 0;
  cl_mem d_A=0, d_B=0, d_C=0;
  cl_device_id devID;
  
  
  int *h_A, *h_B, *h_C;
  h_A = new int[VECSIZE];
  h_B = new int[VECSIZE];
  h_C = new int[VECSIZE];
  
  assert(h_A != NULL &&  h_B != NULL && h_C != NULL);
  for(int i=0;i<VECSIZE; i++)
      h_A[i] = h_B[i] = i;
  
  cl_command_queue_properties qprop;
  setupDevice(cont, q, qprop, devID);

  // create device memory objects
  d_A = clCreateBuffer (cont, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, DATASIZE, h_A, &errNum);
  d_B = clCreateBuffer (cont, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, DATASIZE, h_B, &errNum);
  d_C = clCreateBuffer (cont, CL_MEM_WRITE_ONLY, DATASIZE, NULL, &errNum);

//   errNum = clEnqueueWriteBuffer(q, d_A, CL_TRUE, 0, DATASIZE, h_A, 0, NULL, NULL);
//   errNum = clEnqueueWriteBuffer(q, d_B, CL_TRUE, 0, DATASIZE, h_B, 0, NULL, NULL);

  // read program source and build it
  if(!setupProgramAndKernel(devID, cont, "vectorAdd.cl", pr, "vecAdd", kernel))
  {
      cleanUp (cont, q, pr, kernel, d_A, d_B, d_C);
      return 1;
  }
    
  // setup kernel parameters
  errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
  errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
  errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
  if (isError(errNum, "Failed to set kernel parameters"))
    {
      cleanUp (cont, q, pr, kernel, d_A, d_B, d_C);
      return 1;
    }

  // work item index space and group size setup
  size_t idxSpace[] = { VECSIZE };
  size_t localWorkSize[] = { 64 };

  errNum = clEnqueueNDRangeKernel (q, kernel, 1, NULL, idxSpace, localWorkSize, 0, NULL, NULL);

  errNum = clEnqueueReadBuffer(q, d_C, CL_TRUE, 0, DATASIZE, h_C, 0, NULL, NULL);

  if(!isError(errNum, "Failed to get result vector"))
  {
      for(int i=0;i<VECSIZE;i++)
         cout << h_C[i] << " " ;
      cout << endl;
  }   
  cleanUp (cont, q, pr, kernel, d_A, d_B, d_C);

  delete [] h_A;
  delete [] h_B;
  delete [] h_C;
  
  return 0;
}
