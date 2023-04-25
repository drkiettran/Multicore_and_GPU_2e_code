#pragma once

#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

const int MAXNUMDEV = 10;
/*-------------------------------------*/
// Reads device code from a file and returns a pointer to the buffer holding it
char * readCLFromFile(
         const char* file)  // File name holding the device source code (IN)
{
   FILE *f=fopen(file,"rt");
   if(f==NULL)
   {
      printf("Failed to open file %s for reading\n", file);
      return NULL;
   }
   
   fseek(f, 0, SEEK_END);
   long fsize = ftell(f);
   fseek(f, 0, SEEK_SET);
   
   char *buffer = (char*)malloc(sizeof(char)*(fsize+1));
   if(buffer==NULL)
   {
      printf("Failed to allocate memory for reading file %s\n", file);
      return NULL;
   }
   
   fread(buffer, fsize, 1, f);
   buffer[fsize]=0;
   
   fclose(f);
   return buffer;
}
/*-------------------------------------*/
// Returns false if status if CL_SUCCESS 
bool isError(
            cl_int status,  // Status returned by a OpenCL function  (IN)
            const char *msg)// Message to be printed if status is not CL_SUCCESS (IN)
{
   if(status == CL_SUCCESS)
       return false;
   else
   {
       printf("%s\n",msg);
       return true;
   }
}

/*-------------------------------------*/
void setupDevice(
                 cl_context &cont,   // Reference to context to be created (IN/OUT)
                 cl_command_queue &q,// Reference for queue to be created (IN/OUT)
                 cl_command_queue_properties &qprop, // Array to queue properties (IN/OUT)
                 cl_device_id &id)   // Reference to device to be utilized. Defaults to the first one listed by the clGetPlatformIDs call (IN/OUT)
{
  cl_uint numPlatforms;
  cl_platform_id firstPlatformId;
  cl_device_id devID[MAXNUMDEV];
  cl_uint numDev;
  cl_int errNum;
  
  // Get a reference to an object representing a platform 
  errNum = clGetPlatformIDs (1, &firstPlatformId, &numPlatforms);
  if (isError(errNum,"Failed to find any OpenCL platforms.") || numPlatforms <= 0)
    exit(1);
  
  // Get the device IDs matching the CL_DEVICE_TYPE parameter, up to the MAXNUMDEV limit
  errNum = clGetDeviceIDs (firstPlatformId, CL_DEVICE_TYPE_CPU, MAXNUMDEV, devID, &numDev);
  if (isError(errNum, "Failed to find any OpenCL devices.") || numDev <= 0)
     exit(2);

  char devName[100];
  size_t nameLen;
  for (int i = 0; i < numDev; i++)
    {
      errNum = clGetDeviceInfo (devID[i], CL_DEVICE_NAME, 100, (void *) devName, &nameLen);
      if (errNum == CL_SUCCESS)
        printf("Device %i is %s\n",i,  devName);
    }
    
  cl_context_properties prop[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties) firstPlatformId,
    0                           // termination
  };

  cont = clCreateContext (prop, numDev, devID, NULL,    // no callback function
                          NULL, // no data for callback
                          &errNum);
  if (isError(errNum, "Failed to create a context. "))
    {
        exit(3);
    }

  q = clCreateCommandQueue(cont, devID[0], qprop, &errNum);

  if (isError(errNum, "Failed to create a command queue"))
    {
      clReleaseContext (cont);  
      exit(4);
    }

  // return chosen device ID
  id = devID[0];
}
/*-------------------------------------*/
// Returns true if program was compiled successfully
bool setupProgram(
           cl_device_id &id, // Reference to device to be targetted (IN)
           cl_context &cont, // Reference to context to be utilized (IN)
           const char *programFile, // Pointer to file name holding the source code (IN)
           cl_program &pr)   // Reference to program object to be created (IN/OUT)
{
    cl_int errNum;  
    char *source = readCLFromFile(programFile);

    pr = clCreateProgramWithSource (cont, 1, (const char**)&source, NULL, &errNum);
    if(isError(errNum, "Failed to create program."))
        return false;

    errNum = clBuildProgram (pr, 1, &id, NULL, NULL, NULL);
    if(isError(errNum, "Failed to build program."))
        return false;

    return true;
}
/*-------------------------------------*/
// Returns true if program was compiled successfully
bool setupProgramAndKernel(
         cl_device_id &id, // Reference to device to be targetted (IN)
         cl_context cont,  // Reference to context to be utilized (IN)
         const char *programFile, // Pointer to file name holding the source code (IN)
         cl_program &pr,   // Reference to program object to be created (IN/OUT)
         const char *kernelName, // Pointer to string identifying the kernel function (IN)
         cl_kernel &kernel)//Reference to kernel object to be created (IN/OUT)
{
    cl_int errNum;  
    char *source = readCLFromFile(programFile);
 
    pr = clCreateProgramWithSource (cont, 1, (const char**)&source, NULL, &errNum);
    if(isError(errNum, "Failed to create program."))
        return false;

    errNum = clBuildProgram (pr, 1, &id, NULL, NULL, NULL);
    if(isError(errNum, "Failed to build program."))
        return false;

    kernel = clCreateKernel (pr, kernelName, &errNum);
     if (isError(errNum, "Failed to create kernel."))
        return false;
       
    return true;
}
