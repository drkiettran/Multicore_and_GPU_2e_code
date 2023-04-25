/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : July 2018
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ deviceList.cpp -lOpenCL -o deviceList
 ============================================================================
 */

#include <iostream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

const int MAXNUMDEV=10;
int main()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_device_id devID[MAXNUMDEV];
    cl_uint numDev;
    
    // Get a reference to an object representing a platform 
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        cerr << "Failed to find any OpenCL platforms." << endl;
        return 1;
    }
    
    // Get the device IDs matching the CL_DEVICE_TYPE parameter, up to the MAXNUMDEV limit
    errNum = clGetDeviceIDs( firstPlatformId,CL_DEVICE_TYPE_ALL, MAXNUMDEV, devID, &numDev);
    if (errNum != CL_SUCCESS || numDev <= 0)
    {
        cerr << "Failed to find any OpenCL devices." << endl;
        return 2;
    }
 
    char devName[100];
    size_t nameLen;
    for(int i=0;i<numDev;i++)
    {
      errNum = clGetDeviceInfo(devID[i], CL_DEVICE_NAME, 100, (void*)devName, &nameLen);
      if(errNum == CL_SUCCESS)
          cout << "Device " << i << " is " << devName << endl;
    }
    return 0;   
}
