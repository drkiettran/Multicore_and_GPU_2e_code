/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : January 2020
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : 
 ============================================================================
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

const int MAXNUMDEV = 10;
/*-------------------------------------*/
// Retrieves the compilation errors 
char *getCompilationError (cl_program & pr,     // Reference to program object that failed to compile (IN)
                           cl_device_id & id)   // Target device ID (IN)
{
  size_t logSize;
  // Retrieve message size in "logSize"
  clGetProgramBuildInfo (pr, id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
  // Allocate the necessary buffer space
  char *logMsg = (char *) malloc (logSize);
  // Retrieve the actual error message
  clGetProgramBuildInfo (pr, id, CL_PROGRAM_BUILD_LOG, logSize, logMsg, NULL);
  return logMsg;
}

/*-------------------------------------*/
// Reads device code from a file and returns a pointer to the buffer holding it
char *readCLFromFile (const char *file) // File name holding the device source code (IN)
{
  FILE *f = fopen (file, "rt");
  if (f == NULL)
    {
      printf ("Failed to open file %s for reading\n", file);
      return NULL;
    }

  fseek (f, 0, SEEK_END);
  long fsize = ftell (f);
  fseek (f, 0, SEEK_SET);

  char *buffer = (char *) malloc (sizeof (char) * (fsize + 1));
  if (buffer == NULL)
    {
      printf ("Failed to allocate memory for reading file %s\n", file);
      return NULL;
    }

  fread (buffer, fsize, 1, f);
  buffer[fsize] = 0;

  fclose (f);
  return buffer;
}

/*-------------------------------------*/
// Returns false if status is CL_SUCCESS 
bool isError (cl_int status,    // Status returned by a OpenCL function  (IN)
              const char *msg)  // Message to be printed if status is not CL_SUCCESS (IN)
{
  if (status == CL_SUCCESS)
    return false;
  else
    {
      printf ("%s\n", msg);
      return true;
    }
}

/*-------------------------------------*/
// Calls the cleanup function if status is not CL_SUCCESS, after printing message
void handleError (cl_int status,        // Status returned by a OpenCL function  (IN)
                  const char *msg,      // Message to be printed if status is not CL_SUCCESS (IN)
                  void (*cleanup) ())   // Pointer to cleanup function
{
  if (status != CL_SUCCESS)
    {
      printf ("%s\n", msg);
      (*cleanup) ();
      exit (1);
    }
}

/*-------------------------------------*/
void setupDevice (cl_context & cont,    // Reference to context to be created (IN/OUT)
                  cl_command_queue & q, // Reference for queue to be created (IN/OUT)
                  cl_queue_properties * qprop,  // Array to queue properties (IN/OUT)
                  cl_device_id & id,     // Reference to device to be utilized. Defaults to the first one listed by the clGetPlatformIDs call (IN/OUT)
                  cl_device_type type = CL_DEVICE_TYPE_ALL)
{
  cl_uint numPlatforms;
  cl_platform_id platforms[3]; // up to 3 platforms examined
  cl_device_id devID[MAXNUMDEV];
  cl_uint numDev;
  cl_int errNum;

  // Get a reference to an object representing a platform 
  errNum = clGetPlatformIDs (3, platforms, &numPlatforms);
  if (isError (errNum, "Failed to find any OpenCL platforms.") || numPlatforms <= 0)
    exit (1);

  numPlatforms = std::min((cl_uint)3, numPlatforms);
  
  // Get the device IDs matching the CL_DEVICE_TYPE parameter, up to the MAXNUMDEV limit
  int chosenPlatform=-1;
  for(int i=0;i<numPlatforms;i++)
  {
    errNum = clGetDeviceIDs (platforms[i], type, MAXNUMDEV, devID, &numDev);
    if (errNum == CL_SUCCESS)
    {
        chosenPlatform=i;
        printf("Chosen %i\n",chosenPlatform);
        break;
    }        
  }
  
  if(chosenPlatform == -1)
     {
       printf("Failed to find any OpenCL devices.\n");
       exit (2);
     }

  // print out information about the OpenCL platform and the corresponding devices
  char devName[100];
  size_t nameLen;
  errNum = clGetPlatformInfo (platforms[ chosenPlatform], CL_PLATFORM_VERSION, 100, (void *) devName, &nameLen);        printf ("Platform : %s\n", devName);
  for (int i = 0; i < numDev; i++)
    {
      errNum = clGetDeviceInfo (devID[i], CL_DEVICE_NAME, 100, (void *) devName, &nameLen);
      if (errNum == CL_SUCCESS)
        printf ("Device %i is %s\n", i, devName);
    }

    
  cl_context_properties prop[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties) platforms[chosenPlatform],
    0                           // termination
  };

  cont = clCreateContext (prop, numDev, devID, NULL,    // no callback function
                          NULL, // no data for callback
                          &errNum);
  if (isError (errNum, "Failed to create a context. "))
    {
      exit (3);
    }

  q = clCreateCommandQueueWithProperties (cont, devID[0], qprop, &errNum);

  if (isError (errNum, "Failed to create a command queue"))
    {
      clReleaseContext (cont);
      exit (4);
    }

  // return chosen device ID
  id = devID[0];
}

/*-------------------------------------*/
// Returns true if program was compiled successfully
bool setupProgram (cl_device_id & id,   // Reference to device to be targetted (IN)
                   cl_context & cont,   // Reference to context to be utilized (IN)
                   const char *programFile,     // Pointer to file name holding the source code (IN)
                   cl_program & pr)     // Reference to program object to be created (IN/OUT)
{
  cl_int errNum;
  char *source = readCLFromFile (programFile);

  pr = clCreateProgramWithSource (cont, 1, (const char **) &source, NULL, &errNum);
  if (isError (errNum, "Failed to create program."))
    return false;

  char options[]="-cl-std=CL2.0";
  errNum = clBuildProgram (pr, 1, &id, options, NULL, NULL);
  if (isError (errNum, "Failed to build program."))
    return false;

  return true;
}

/*-------------------------------------*/
// Returns true if program was compiled successfully
bool setupProgramAndKernel (cl_device_id & id,  // Reference to device to be targetted (IN)
                            cl_context cont,    // Reference to context to be utilized (IN)
                            const char *programFile,    // Pointer to file name holding the source code (IN)
                            cl_program & pr,    // Reference to program object to be created (IN/OUT)
                            const char *kernelName,     // Pointer to string identifying the kernel function (IN)
                            cl_kernel & kernel) //Reference to kernel object to be created (IN/OUT)
{
  cl_int errNum;
  char *source = readCLFromFile (programFile);

  pr = clCreateProgramWithSource (cont, 1, (const char **) &source, NULL, &errNum);
  if (isError (errNum, "Failed to create program."))
    return false;

  char options[]="-cl-std=CL2.0";
  errNum = clBuildProgram (pr, 1, &id, options, NULL, NULL);
  if (isError (errNum, "Failed to build program."))
    return false;

  kernel = clCreateKernel (pr, kernelName, &errNum);
  if (isError (errNum, "Failed to create kernel."))
    return false;

  return true;
}
