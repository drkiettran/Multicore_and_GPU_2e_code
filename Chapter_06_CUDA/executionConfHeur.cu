/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.1
 Last modified : November 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc executionConfHeur.cu -o executionConfHeur
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>

//-------------------------------------------------------
void calcExecConf (int numberOfThreads, int registersPerThread, int sharedPerThread, int &bestThreadsPerBlock, int &bestTotalBlocks)
{
  cudaDeviceProp pr;
  cudaGetDeviceProperties (&pr, 0);     // replace 0 with appropriate ID in case of a multi-GPU system

  int maxRegs = pr.regsPerBlock;
  int SM = pr.multiProcessorCount;
  int warp = pr.warpSize;
  int sharedMem = pr.sharedMemPerBlock;
  int maxThreadsPerSM = pr.maxThreadsPerMultiProcessor;
  int totalBlocks;
  float imbalance, bestimbalance;
  int threadsPerBlock;
  int numWarpSchedulers = 4;

  bestimbalance = SM;

  // initially calculate the maximum possible threads per block. Incorporate limits imposed by :
  // 1) SM hardware 
  threadsPerBlock = maxThreadsPerSM;
  // 2) registers
  threadsPerBlock = min (threadsPerBlock, maxRegs / registersPerThread);
  // 3) shared memory size
  threadsPerBlock = min (threadsPerBlock, sharedMem / sharedPerThread);
  
  // make sure it is a multiple of warpSize  
  int tmp = threadsPerBlock / warp;
  threadsPerBlock = (tmp+1) * warp;

  for (; threadsPerBlock >= numWarpSchedulers * warp && bestimbalance != 0; threadsPerBlock -= warp)
    {
      totalBlocks = (int) ceil (1.0 * numberOfThreads / threadsPerBlock);

      if (totalBlocks % SM == 0)
        imbalance = 0;
      else
        {
          int blocksPerSM = totalBlocks / SM;   // some SMs get this number and others get +1 block
          imbalance = (SM - (totalBlocks % SM)) / (blocksPerSM + 1.0);
        }
       printf ("%i %i %lf %lf\n", threadsPerBlock, totalBlocks, imbalance, bestimbalance);
      if (bestimbalance >= imbalance)
        {
          bestimbalance = imbalance;
          bestThreadsPerBlock = threadsPerBlock;
          bestTotalBlocks = totalBlocks;
        }
    }
}

//-------------------------------------------------------
int main (int argc, char **argv)
{
  int registersPerThread;
  int sharedPerThread;
  int numberOfThreads;
  int bestThreadsPerBlock, bestTotalBlocks;

  numberOfThreads = atoi (argv[1]);
  registersPerThread = atoi (argv[2]);
  sharedPerThread = atoi (argv[3]);

  calcExecConf (numberOfThreads, registersPerThread, sharedPerThread, bestThreadsPerBlock, bestTotalBlocks);
  printf ("BEST grid with %i blocks, each with %i threads\n", bestTotalBlocks, bestThreadsPerBlock);
  return 1;
}
