/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : May 2020
 License       : Released under the GNU GPL 3.0
 Description   : Row-wise partitioning
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
#include <thread>
#include <memory>
#include <mutex>

#include "../cl_utility.h"

using namespace std;
using namespace std::chrono;

#include "const.h"

//************************************************************
struct BlockDescr
{
  double upperX, upperY, stepX, stepY;
  int width, height;
  int startRow;
};
//************************************************************
const int DEVICEALLOC = 64;

class LoadMonitor
{
private:
  mutex l;
  int width, height;
  double upperX, upperY;
  int currentRow = 0;
  double xStep, yStep;
public:
  LoadMonitor (double uX, double uY, double lX, double lY, int w, int h):upperX (uX), upperY (uY), width (w), height (h)
  {
    xStep = (lX - upperX) / width;
    yStep = (upperY - lY) / height;
  }
  bool getNextBlock (BlockDescr & d);
};
//------------------------------------------
// returns true if the d structure is populated. False if work is done
bool LoadMonitor::getNextBlock (BlockDescr & d)
{
  lock_guard < mutex > lg (l);

  if (currentRow == height)
    return false;

  d.upperY = upperY + yStep * currentRow;
  d.upperX = upperX;
  d.stepX = xStep;
  d.stepY = yStep;
  d.width = width;
  d.startRow = currentRow;
  int toAlloc = (height - currentRow >= DEVICEALLOC) ? DEVICEALLOC : height - currentRow;       
  currentRow+=toAlloc;
  d.height = toAlloc;
  return true;
}

//************************************************************
class HostFE
{
private:    
  cl_context cont = 0;     // initialize for cleanup check
  cl_command_queue q = 0;
  cl_program pr = 0;
  cl_kernel kern = 0;
  bool OCL2support = true;
  unsigned char *data = 0; // image data
  cl_mem d_data = 0;       // image data if OpenCL 2.0 is not supported
  shared_ptr < QImage > img;
  shared_ptr < LoadMonitor > mon;
  cl_device_type t;
  
public:
  int rowsProcessed=0;
  HostFE(shared_ptr < QImage > img, shared_ptr < LoadMonitor > mon, cl_device_type t)
  {
      this->img=img;
      this->mon=mon;
      this->t=t;
  }
  void operator()();
  void cleanUp();    
};

//--------------------------------------
void HostFE::cleanUp ()
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
void HostFE::operator()()
{
  BlockDescr work;
  cl_int errNum;
  cl_device_id devID;
  cl_event kernComplete;

  // context and queue setup
  setupDevice (cont, q, NULL, devID, t);

  //------------------------------------------
  // read program source and build it
  if (!setupProgramAndKernel (devID, cont, "kernel.cl", pr, "mandelKernel", kern))
    {
      if (pr != 0)
        printf ("Error: %s\n", getCompilationError (pr, devID));
      cleanUp ();
      return;
    }

  // check if SVM is supported
  char attr[20];
  size_t attrLen;
  clGetDeviceInfo (devID, CL_DEVICE_VERSION, 20, (void *) attr, &attrLen);
  if (attr[7] == '1')           // Character at position 7 is the OpenCL major version
    OCL2support = false;


  int threadX, threadY;

  //------------------------------------------
  // main thread loop
  while (mon->getNextBlock (work))
    {
      rowsProcessed+=work.height;
      // memory allocation and some initializations done only during the first iteration       
      if (data == 0)
        {
          if (OCL2support)
            data = (unsigned char *) clSVMAlloc (cont, CL_MEM_READ_WRITE, work.width * work.height * sizeof (unsigned char), 0);
          else
            {
              d_data = clCreateBuffer (cont, CL_MEM_READ_WRITE, work.width * work.height * sizeof (unsigned char), NULL, &errNum);
              data = new unsigned char[work.width * work.height];
            }
          threadX = (int) ceil (work.width * 1.0 / THR_BLK_X);
          threadY = (int) ceil (work.height * 1.0 / THR_BLK_Y);
          threadX = ((threadX + BLOCK_SIDE - 1) / BLOCK_SIDE) * BLOCK_SIDE;     // make sure that a work item divides the idxSpace evenly
          threadY = ((threadY + BLOCK_SIDE - 1) / BLOCK_SIDE) * BLOCK_SIDE;
        }


      if (OCL2support)
        errNum = clSetKernelArgSVMPointer (kern, 0, data);
      else
        errNum = clSetKernelArg (kern, 0, sizeof (cl_mem), (void *) &d_data);

      errNum |= clSetKernelArg (kern, 1, sizeof (double), (void *) &work.upperX);
      errNum |= clSetKernelArg (kern, 2, sizeof (double), (void *) &work.upperY);
      errNum |= clSetKernelArg (kern, 3, sizeof (double), (void *) &work.stepX);
      errNum |= clSetKernelArg (kern, 4, sizeof (double), (void *) &work.stepY);
      errNum |= clSetKernelArg (kern, 5, sizeof (int), (void *) &work.width);
      errNum |= clSetKernelArg (kern, 6, sizeof (int), (void *) &work.height);
      if (isError (errNum, "Failed to set kernel parameters"))
        {
          cleanUp ();
          return ;
        }

      size_t idxSpace[] = { threadX, threadY };
      size_t localWorkSize[] = { BLOCK_SIDE, BLOCK_SIDE };

      // launch GPU kernel
      errNum = clEnqueueNDRangeKernel (q, kern, 2, NULL, idxSpace, localWorkSize, 0, NULL, &kernComplete);
      if (isError (errNum, "Failed to launch kernel"))
        {
          cleanUp ();
          return;
        }

      if (!OCL2support)
        errNum = clEnqueueReadBuffer (q, d_data, CL_TRUE, 0, work.width * work.height * sizeof (unsigned char), data, 0, NULL, NULL);
      else
        {
          //------------------------------------------
          // wait for the kernel to finish
          cl_event evlist[] = { kernComplete };
          errNum = clWaitForEvents (1, evlist);
          if (isError (errNum, "Failed to wait for event"))
            {
              cleanUp ();
              return ;
            }
        }

      //------------------------------------------
      //copy results into QImage object   
      for (int j = 0; j < work.height; j++)
        for (int i = 0; i < work.width; i++)
          {
            int color = data[j * work.width + i];
            img->setPixel (i, j + work.startRow, qRgb (256 - color, 256 - color, 256 - color));
          }
    }
  // clean-up allocated memory
  cerr << t << " Processed " << rowsProcessed << " rows\n";
  cleanUp ();

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

  shared_ptr < QImage > img = make_shared < QImage > (imgX, imgY, QImage::Format_RGB32);
  shared_ptr < LoadMonitor > mon = make_shared < LoadMonitor > (upperCornerX, upperCornerY, lowerCornerX, lowerCornerY, imgX, imgY);

  // create a thread for handling the host as an OpenCL device  
  thread hostThr(std::move(HostFE(img, mon, CL_DEVICE_TYPE_CPU)));

  HostFE gpu(img, mon, CL_DEVICE_TYPE_GPU);
  gpu(); // use the master thread to control the GPU device

  // wait for CPU handling thread to finish
  hostThr.join ();
  double computeTime = t.elapsed ();
  img->save ("mandel.png", "PNG", 0);

  cout << "Compute (ms):" << computeTime << " Total (ms):" << t.elapsed () << endl;
  return 0;
}
