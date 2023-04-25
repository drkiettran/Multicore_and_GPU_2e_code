/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : February 2020
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : qmake; make
 ============================================================================
 */
#include <QImage>
#include <QRgb>
#include <QTime>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <chrono>

#include "../cl_utility.h"

using namespace std;
using namespace std::chrono;

#include "const.h"

static cl_context cont = 0;     // initialize for cleanup check
static cl_command_queue q = 0;
static cl_program pr = 0;
static cl_kernel kern = 0;
static bool OCL2support = true;
static unsigned char *data = 0; // image data
static cl_mem d_data = 0;       // image data if OpenCL 2.0 is not supported

//************************************************************
void cleanUp ()
{
  if (kern != 0)
    clReleaseKernel (kern);

  if (pr != 0)
    clReleaseProgram (pr);

  if (q != 0)
    clReleaseCommandQueue (q);

  if (cont != 0)
    {
      if (OCL2support)
        clSVMFree (cont, data);
      else
        clReleaseMemObject (d_data);
      clReleaseContext (cont);
    }
}

//************************************************************
// Host front-end function that allocates the memory and launches the GPU kernel
double hostFE (double upperX, double upperY, double lowerX, double lowerY, QImage * img, int resX, int resY)
{
  auto t1 = high_resolution_clock::now ();

  cl_int errNum;
  cl_device_id devID;
  cl_event kernComplete;

  // context and queue setup
  setupDevice (cont, q, NULL, devID, CL_DEVICE_TYPE_GPU);

  //------------------------------------------
  // read program source and build it
  if (!setupProgramAndKernel (devID, cont, "kernel.cl", pr, "mandelKernel", kern))
    {
      if (pr != 0)
        printf ("Error: %s\n", getCompilationError (pr, devID));
      cleanUp ();
      return 1;
    }

  // check if SVM is supported
  char attr[20];
  size_t attrLen;
  clGetDeviceInfo (devID, CL_DEVICE_VERSION, 20, (void *) attr, &attrLen);
  if (attr[7] == '1')           // Character at position 7 is the OpenCL major version
    OCL2support = false;

  if (OCL2support)
    data = (unsigned char *) clSVMAlloc (cont, CL_MEM_READ_WRITE, resX * resY * sizeof (unsigned char), 0);
  else
    {
      d_data = clCreateBuffer (cont, CL_MEM_READ_WRITE, resX * resY * sizeof (unsigned char), NULL, &errNum);
      data = new unsigned char[resX * resY];
    }

  double stepX = (lowerX - upperX) / resX;
  double stepY = (upperY - lowerY) / resY;

  if (OCL2support)
    errNum = clSetKernelArgSVMPointer (kern, 0, data);
  else
    errNum = clSetKernelArg (kern, 0, sizeof (cl_mem), (void *) &d_data);

  errNum |= clSetKernelArg (kern, 1, sizeof (double), (void *) &upperX);
  errNum |= clSetKernelArg (kern, 2, sizeof (double), (void *) &upperY);
  errNum |= clSetKernelArg (kern, 3, sizeof (double), (void *) &stepX);
  errNum |= clSetKernelArg (kern, 4, sizeof (double), (void *) &stepY);
  errNum |= clSetKernelArg (kern, 5, sizeof (int), (void *) &resX);
  errNum |= clSetKernelArg (kern, 6, sizeof (int), (void *) &resY);
  if (isError (errNum, "Failed to set kernel parameters"))
    {
      cleanUp ();
      return 1;
    }

  int threadX, threadY;
  threadX = (int) ceil (resX * 1.0 / THR_BLK_X);
  threadY = (int) ceil (resY * 1.0 / THR_BLK_Y);
  threadX = ((threadX + BLOCK_SIDE - 1) / BLOCK_SIDE) * BLOCK_SIDE;     // make sure that a work item divides the idxSpace evenly
  threadY = ((threadY + BLOCK_SIDE - 1) / BLOCK_SIDE) * BLOCK_SIDE;

  size_t idxSpace[] = { threadX, threadY };
  size_t localWorkSize[] = { BLOCK_SIDE, BLOCK_SIDE };

  // launch GPU kernel
  errNum = clEnqueueNDRangeKernel (q, kern, 2, NULL, idxSpace, localWorkSize, 0, NULL, &kernComplete);
  if (isError (errNum, "Failed to launch kernel"))
    {
      cleanUp ();
      return 1;
    }

  if (!OCL2support)
    errNum = clEnqueueReadBuffer (q, d_data, CL_TRUE, 0, resX * resY * sizeof (unsigned char), data, 0, NULL, NULL);
  else
    {

      //------------------------------------------
      // wait for the kernel to finish
      cl_event evlist[] = { kernComplete };
      errNum = clWaitForEvents (1, evlist);
      if (isError (errNum, "Failed to wait for event"))
        {
          cleanUp ();
          return 1;
        }
    }

  //copy results into QImage object   
  for (int j = 0; j < resY; j++)
    for (int i = 0; i < resX; i++)
      {
        int color = data[j * resX + i];
        img->setPixel (i, j, qRgb (256 - color, 256 - color, 256 - color));
      }

  // clean-up allocated memory
  cleanUp ();
  auto t2 = high_resolution_clock::now ();
  auto dur = t2 - t1;
  return duration_cast < milliseconds > (dur).count ();
}

//************************************************************

int main (int argc, char *argv[])
{
  double upperCornerX, upperCornerY;
  double lowerCornerX, lowerCornerY;

  QTime t;
  t.start ();

  upperCornerX = atof (argv[1]);
  upperCornerY = atof (argv[2]);
  lowerCornerX = atof (argv[3]);
  lowerCornerY = atof (argv[4]);

  int imgX = 3840, imgY = 2160;
  if (argc > 6)
    {
      imgX = atoi (argv[5]);
      imgY = atoi (argv[6]);
    }

  QImage *img = new QImage (imgX, imgY, QImage::Format_RGB32);

  double gpuOpsTime = hostFE (upperCornerX, upperCornerY, lowerCornerX, lowerCornerY, img, imgX, imgY);

  double computeTime = t.elapsed ();
  img->save ("mandel.png", "PNG", 0);

  cout << "Compute (ms):" << computeTime << " Total (ms):" << t.elapsed () << " GPU time: " << gpuOpsTime << endl;
  return 0;
}
