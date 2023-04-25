/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2019
 License       : Released under the GNU GPL 3.0
 Description   : Image convolution in OpenCL C host program using cl_utility.h
 To build use  : icc -lOpenCL image_conv.cpp -o image_conv
                 OR
                 clang++ -lOpenCL image_conv.cpp -o image_conv
 ============================================================================
 */

#include <iostream>
#include <string>

#include "cl_utility.h"
#include <CL/cl2.hpp>
#include "common/pgm.cpp"

using namespace std;

static cl_context cont = 0;          // initialize for cleanup check
static cl_command_queue q = 0;
static cl_program pr = 0;
static cl_kernel kernel = 0;
static cl_mem d_inImg = 0, d_outImg = 0, d_filter = 0;

//============================================
void cleanUp()
{
  if (d_inImg != 0)
    clReleaseMemObject (d_inImg);

  if (d_outImg != 0)
    clReleaseMemObject (d_outImg);

  if (d_filter != 0)
    clReleaseMemObject (d_filter);

  if (kernel != 0)
    clReleaseKernel (kernel);

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
  cl_event filtTrans, imgTrans, kernComplete;

  setupDevice (cont, q, NULL, devID);

  //------------------------------------------
  // read program source and build it
  if (!setupProgramAndKernel (devID, cont, "imageConv.cl", pr, "imageConv", kernel))
    {
      cleanUp();
      return 1;
    }

  //------------------------------------------
  // host-side memory allocation & I/O
  PGMImage inImg (argv[1]);

  float filter[] = { 1.0 / 16, 2.0 / 16, 1.0 / 16,
    2.0 / 16, 4.0 / 16, 2.0 / 16,
    1.0 / 16, 2.0 / 16, 1.0 / 16
  };
  int filterSize = 3;

  //------------------------------------------
  // allocate memory on the target device and copy data from the host to it

  // image specification
  cl_image_desc imgDescr;
  imgDescr.image_type = CL_MEM_OBJECT_IMAGE2D;
  imgDescr.image_width = inImg.x_dim;
  imgDescr.image_height = inImg.y_dim;
  imgDescr.image_row_pitch = 0;
  imgDescr.num_mip_levels = 0;
  imgDescr.num_samples = 0;
  imgDescr.mem_object = NULL;

  cl_image_format imgFmr;
  imgFmr.image_channel_order = CL_R;
  imgFmr.image_channel_data_type = CL_UNSIGNED_INT8;

  // Input image
  d_inImg = clCreateImage (cont, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &imgFmr, &imgDescr, inImg.pixels, &errNum);
  if (isError (errNum, "Failed to create device input image"))
    {
      cleanUp();
      return 1;
    }

  // Output image
  d_outImg = clCreateImage (cont, CL_MEM_WRITE_ONLY, &imgFmr, &imgDescr, NULL, &errNum);
  if (isError (errNum, "Failed to create device output image"))
    {
      cleanUp();
      return 1;
    }

  // Device filter
  d_filter = clCreateBuffer (cont, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 9 * sizeof (float), &(filter[0]), &errNum);
  if (isError (errNum, "Failed to create device filter buffer"))
    {
      cleanUp();
      return 1;
    }

  // Image sampler  
  cl_sampler_properties prop[] = { CL_SAMPLER_NORMALIZED_COORDS, CL_FALSE, 0 };
  cl_sampler sampler = clCreateSamplerWithProperties (cont, prop, &errNum);
  if (isError (errNum, "Failed to create sampler"))
    {
      cleanUp();
      return 1;
    }

  //------------------------------------------
  // specify the parameters that will be used for the kernel execution
  // setup the work item index space and group size
  size_t idxSpace[] = { (size_t) inImg.x_dim, (size_t) inImg.y_dim };
  size_t localWorkSize[] = { 16, 16 };

  // enqueue kernel execution request
  errNum = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *) &d_inImg);
  errNum |= clSetKernelArg (kernel, 1, sizeof (int), (void *) &(inImg.x_dim));
  errNum |= clSetKernelArg (kernel, 2, sizeof (int), (void *) &(inImg.y_dim));
  errNum |= clSetKernelArg (kernel, 3, sizeof (cl_mem), (void *) &d_filter);
  errNum |= clSetKernelArg (kernel, 4, sizeof (int), (void *) &filterSize);
  errNum |= clSetKernelArg (kernel, 5, sizeof (cl_mem), (void *) &d_outImg);
  errNum |= clSetKernelArg (kernel, 6, sizeof (cl_sampler), (void *) &sampler);
  if (isError (errNum, "Failed to set kernel parameters"))
    {
      cleanUp();
      return 1;
    }

  errNum = clEnqueueNDRangeKernel (q, kernel, 2, NULL, idxSpace, localWorkSize, 0, NULL, &kernComplete);
  if (isError (errNum, "Failed to launch kernel"))
    {
      cleanUp();
      return 1;
    }

  //------------------------------------------
  // collect the results by copying them from the device to the host memory
  size_t origin[] = { 0, 0, 0 };
  size_t region[] = { (size_t) inImg.x_dim, (size_t) inImg.y_dim, 1 };
  cl_event waitList[] = { kernComplete };
  clEnqueueReadImage (q, d_outImg, CL_TRUE, origin, region, 0, 0, inImg.pixels, 1, waitList, NULL);

  inImg.write (argv[2]);

  // release resources
  cleanUp();

  return 0;
}

