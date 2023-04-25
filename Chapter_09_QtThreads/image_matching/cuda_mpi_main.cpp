// CUDA with MPI version, based on the Rapidmind version gpu-mpi.cpp
// June 2007, G. Barlas
// Jan 2008, getSpeed() is customized to work for 1 GPU, multiple Pentium 4 cluster
// April 2009, CUDA added

// This is the coordination/communication part of the software

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <fstream>
#include <mpi.h>

#include "common_defs.h"

using namespace std;
//using namespace mpich;

double getSpeed();

//***********************************************************
/*  Fine timing support */
__inline__ unsigned long long int rdtsc ()
{
  unsigned long long int x;
  __asm__ volatile (".byte 0x0f, 0x31":"=A" (x));
  return x;
}
//**********************************************************
int comp(const void* a, const void* b)
{
   nodeLoad *x=(nodeLoad *)a;
   nodeLoad *y=(nodeLoad *)b;
   return (int)(x->left - y->left);
}
//***********************************************************
double partition_old(double *p, int N, int L, int b, int d, double l, double *part, int *quant)
{
  double t;
  double *frac;
  double *prod;

  frac= new double[N];
  prod= new double[N];

  for(int i=1;i<N;i++)
     frac[i] = p[i-1]/(p[i]+l);

  prod[0]=1;
  for(int i=1;i<N;i++)
     prod[i] = prod[i-1]*frac[i];

  double nomin=0, denom=1;
  for(int i=1;i<N;i++)
     denom += prod[i]; 
  for(int i=1;i<N;i++)
     for(int k=1; k<=i; k++)
        nomin += prod[i-1]/prod[k-1] /(p[k]+l);
  nomin *= -l*(d-b)/L;
  nomin ++;

  part[0] = nomin/denom;
  for(int i=1;i<N;i++)
    part[i] = part[i-1] * frac[i] + l*(d-b)/(L*(p[i]+l));

  t = l*(part[0]*L+b) + p[0]*part[0]*L+N*l*d;

  //********************************************************************
  // quantize load
  // assume that the number of images is L/b and that b is the average image size
  int numImg=L/b;
  long diff=0;
  nodeLoad *v= new nodeLoad[N];

  for(int i=0;i<N;i++)
    {
      quant[i] =(int) floor(part[i]*L / b);
      v[i].ID = i;
      v[i].left = (part[i] - quant[i]*b/L)/part[i];
      diff += quant[i];
    }
  diff = numImg - diff; // remaining images
  
  qsort(v, N, sizeof(nodeLoad), comp);

  for(int i=0;i<diff;i++)
    quant[v[i].ID] ++;
  //********************************************************************

  delete[] v;
  delete[] frac;
  delete[] prod;
  return t;
}

//***********************************************************
double partition(double *p, int N, int L, int b, int d, double l, double *part, int *quant)
{
  double t, sum=1;
  double *frac;

  frac= new double[N];
  frac[0]=1;
  for(int i=1;i<N;i++)
    {
     frac[i] = (p[0]+l)/(p[i]+l);
     sum+=frac[i];
    }

  part[0] = 1/sum;
  cerr << "PART0 " << part[0] << endl;

  for(int i=1;i<N;i++)
    part[i] = part[0] * frac[i];

  t = l*(part[0]*L+b) + p[0]*part[0]*L+ l*d;

  //-----------------------------------------------------------------------
  // quantize load
  // assume that the number of images is L/b and that b is the average image size
  int numImg=L/b;
  long diff=0;
  nodeLoad *v= new nodeLoad[N];

  for(int i=0;i<N;i++)
    {
      quant[i] =(int) floor(part[i]*L / b);
      v[i].ID = i;
      v[i].left = (part[i] - quant[i]*b/L)/part[i];
      diff += quant[i];
    }
  diff = numImg - diff; // remaining images

/*  
  qsort(v, N, sizeof(nodeLoad), comp);

  for(int i=0;i<diff;i++)
    quant[v[i].ID] ++;*/

  // another approach to quantization: give load to the least disturned node
  for(int i=0;i<diff;i++)
    {
      int best=0;
      double best_deviation=(-part[0] + (quant[0]+1.0)*b/L)/part[0];
      for(int j=1;j<N;j++)
       {
         double aux = (-part[j] + (quant[j]+1.0)*b/L)/part[j];
         if(aux < best_deviation)
           {
             best_deviation = aux;
             best = j;
           }
       }
      cout << "Best is " << best << " " << best_deviation << endl;
      quant[best]++;
    }
  for(int i=0;i<N;i++)
    cerr << "==== Node " << i << " is getting " << quant[i] << " images for its " << part[i] << " part " << endl;
  //-----------------------------------------------------------------------

  delete[] v;
  delete[] frac;
  return t;
}
//***********************************************************
int main(int argc, char* argv[])
{
   int myid, num_proc;
   double nodeSpeed, *allspeeds;
   int numImages;
   int limits[2];  // for holding the starting and ending image numbers
   int *limitsDist;  // for scattering the limits
   double stTime;
   ImgRegResults *rr;

   MPI::Status status;

   MPI::Init(argc, argv);

   // get essential info
   stTime = MPI::Wtime();
   myid = MPI::COMM_WORLD.Get_rank ();
   num_proc = MPI::COMM_WORLD.Get_size ();
   numImages = atoi(argv[1]);


   rr=new ImgRegResults[num_proc];

   //collect node speed and run partitioning code
   nodeSpeed = getSpeed();
   if(myid==0)
    {
      allspeeds = new double[num_proc];
      limitsDist = new int[num_proc*2];
    }

// cout << myid << " " << nodeSpeed << endl;
   MPI::COMM_WORLD.Gather( &nodeSpeed, 1, MPI::DOUBLE, allspeeds, num_proc, MPI::DOUBLE, 0);

   if(myid==0)
      {
//         for(int i=1;i<num_proc;i++)
//            MPI::COMM_WORLD.Recv( allspeeds+i, 1, MPI::DOUBLE, i, SPEEDTAG, status);

//         allspeeds[0] = nodeSpeed;

         for(int i=0;i<num_proc;i++)
            cout << "Slope of node " << i << " is : " << allspeeds[i] << endl;

        double part[num_proc];
        int imgcount[num_proc];
        double l = 1.0 * num_proc /( 4103.8*1024*1024.0); // average regardless of number of nodes
        partition(allspeeds, num_proc, numImages * IMGSIZE,IMGSIZE,20,l, part, imgcount);
        limitsDist[0] = 0;
        limitsDist[1] = imgcount[0]-1;
        for(int i=1;i<num_proc;i++)
          {
            limitsDist[2*i] = limitsDist[2*i-1]+1;
            limitsDist[2*i+1] = limitsDist[2*i] + imgcount[i]-1;
          }

// for(int i=0;i< 2*num_proc; i++)
//    cout << limitsDist[i] << " ";
// cout << endl;

        limits[0] = limitsDist[0];  // just for node 0
        limits[1] = limitsDist[1];

        // send the limits to the rest
        for(int i=1;i<num_proc;i++)
           MPI::COMM_WORLD.Send( limitsDist+2*i, 2, MPI::INT, i, ASSTAG);
      }
    else
      {
//         MPI::COMM_WORLD.Send( &nodeSpeed, 1, MPI::DOUBLE, 0, SPEEDTAG);
        MPI::COMM_WORLD.Recv( limits, 2, MPI::INT, 0, ASSTAG, status);
      }


//    MPI::COMM_WORLD.Scatter( limitsDist, 2*num_proc, MPI::INT, limits, 2, MPI::INT, 0);
//    cout << "Node " << myid << " is assigned from " << limits[0] << " to " << limits[1] << endl;
   int bestidx = process(limits[0], limits[1]);

   // determine best image
   if(myid==0)
      {
        ImgRegResults localrr[num_proc];
        localrr[0]=rr[bestidx];
        for(int i=1;i<num_proc;i++)
           MPI::COMM_WORLD.Recv( localrr+i, sizeof(ImgRegResults), MPI::CHAR, i, RESTAG, status);

        bestidx=0;
        for(int i=1;i<num_proc;i++)
          if(localrr[bestidx].M < localrr[i].M)
             bestidx = i;
        cout << "Best image match " << localrr[bestidx].ID << endl;
      }
    else
        MPI::COMM_WORLD.Send( rr+bestidx, sizeof(ImgRegResults), MPI::CHAR, 0, RESTAG); // bad hack to avoid data type registration, since all the machine have the same architecture

   // cleanup
   if(myid==0)
    {
      delete[] allspeeds;
      delete[] limitsDist;
      cout << MPI::Wtime() - stTime << endl;
    }
   delete[] rr; 

   MPI::Finalize();
   return 0;
}
