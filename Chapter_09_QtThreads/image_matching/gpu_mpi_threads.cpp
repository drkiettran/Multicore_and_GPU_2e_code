// MPI version
// June 2007, G. Barlas
// Multithreaded version

#include <rapidmind/platform.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <pthread.h>
#include <semaphore.h>

using namespace std;
// using namespace mpi;
using namespace rapidmind;

#define MAXPIXELMEM 1024*1024
#define IMGSIZE 786448
#define IMGBUFFSIZE 5

#define SPEEDTAG 0
#define ASSTAG 1
#define RESTAG 2
//***********************************************************
struct ImgRegResults
{
  int   ID;
  float D1;
  float D2;
  float C;
  float M;
};
ImgRegResults *rr; // registration results
//***********************************************************
union pixelData
{
  unsigned char *cp;
  unsigned short *sp;
  unsigned int *ip;
};
//***********************************************************
class Image  // made into a class to simplify memory management
{
  public:
  int width;
  int height;
  int levels;
  pixelData pixel;

  Image()
  {
    pixel.ip = new unsigned int[MAXPIXELMEM];
  }

  ~Image()
  {
    delete[] pixel.ip;
  }
};
//**********************************************************
struct nodeLoad
{
  int ID;
  double left;
};
//**********************************************************
int comp(const void* a, const void* b)
{
   nodeLoad *x=(nodeLoad *)a;
   nodeLoad *y=(nodeLoad *)b;
   return (int)(x->left - y->left);
}
//***********************************************************
// CPU based calculation
float mutual( unsigned int *img, unsigned int *img2,int numPixels, int levels)
{
  float mutual=0;
  float pi[levels];
  float pj[levels];
  float pij[levels][levels];

  memset(pi,0,levels*sizeof(float));
  memset(pj,0,levels*sizeof(float));
  memset(pij,0,levels*levels*sizeof(float));

  for(int i=0;i<numPixels;i++)
    pi[ img[i]]++;

  for(int i=0;i<numPixels;i++)
    pj[ img2[i]]++;
  
  for(int i=0;i<numPixels;i++)
    pij[ img[i]][img2[i]]++;

  for(int i=0;i<levels;i++)
    for(int j=0;j<levels;j++)
      if(pi[i]!=0 && pj[j]!=0 && pij[i][j]!=0)
        mutual+= pij[i][j] * log(pij[i][j]*numPixels/(pi[i]*pj[j]));
  return mutual/numPixels;
}
//***********************************************************
// Using the GPU
float mutualInfo(Array<1, Value1f> *pa, Array<1, Value1f> *pb, Array<2, Value1f> *pab, int levels)
{
   Array<2, Value1f> logfrac(levels, levels);
   Array<2, Value1f> papb(levels, levels); // product of pa and pb

   // Calculate joint and individual probability distr.
   const float *proba  = pa->read_data();
   const float *probb  = pb->read_data();
   float *probprod  = papb.write_data();
   float *probab = pab->write_data();

   for(int i=0;i<levels;i++)
      for(int j=0;j<levels;j++)
           probprod[i*levels + j] = proba[i] * probb[j];

   for(int i=0;i<levels;i++)
     for(int j=0;j<levels;j++)
       if(probab[j*levels + i]==0)  // make sure the logarithm computation does not cause problems
         {
            probab[j*levels + i]=1;
            probprod[j*levels + i]=1;
         }

    Program entropy = RM_BEGIN {
       In<Value1f> a;  // first input
       In<Value1f> ab;  // second input
       Out<Value1f> r; // output

       r = ab * log( ab/a ); // operation on the data
     } RM_END

  logfrac = entropy(papb, *pab);
  return sum(logfrac).get_value(0) / log(2);  // sum and convert to base-2 logarithm
}
//***********************************************************
Value2f Diff(Array<2, Value1f> *p, Array<2, Value1f> &d)
{
    Array<2, Value2f> res;

    Program diffprod = RM_BEGIN {
       In<Value1f> a;  // first input
       In<Value1f> b;  // second input
       Out<Value2f> r; // output

       r(0) = a * b; 
       r(1) = r(0) * b;
     } RM_END

  res = diffprod(*p, d);
  return sum(res);
}
//***********************************************************
void readImage(Image &img, char *fname)
{
  FILE *fin;
  fin=fopen(fname,"rb");
  fscanf(fin, "%*s%i%i%i",&(img.width), &(img.height), &(img.levels));
  if(img.levels<256)
     fread(img.pixel.cp, sizeof(unsigned char), img.width*img.height, fin);
  else if(img.levels<65536)
     fread(img.pixel.sp, sizeof(unsigned short), img.width*img.height, fin);
  else
     fread(img.pixel.ip, sizeof(unsigned int), img.width*img.height, fin);
  img.levels++;
  fclose(fin);
}
//***********************************************************
void CalcProb(Array<1, Value1f> *p, unsigned int *img, int numPixels, int levels)
{
  float *prob  = p->write_data();
  memset((void *)prob,0,levels*sizeof(float));
  for(int i=0;i<numPixels;i++)
    prob[ img[i]]++;
 
  for(int i=0;i<levels;i++)
    prob[i] /= numPixels;
}
//***********************************************************
void CalcJointProb(Array<2, Value1f> *p, unsigned int *img, unsigned int *img2, int numPixels, int levels)
{
  float *prob  = p->write_data();
  memset((void *)prob,0,levels*levels*sizeof(float));
  for(int i=0;i<numPixels;i++)
    prob[ img[i]*levels + img2[i] ]++;

  for(int i=0;i<levels*levels;i++)
    prob[i] /= numPixels;
}
//***********************************************************
float Covariance(Array<2, Value1ui> *a, Array<2, Value1ui> *b, int numPixels)
{
   Value1f avga;
   Value1f avgb;
   Value1f covaa, covbb, covab, aux;
   Array<2, Value1f> res;

   avga = sum( *a );
   avga /= (1.0*numPixels);
   avgb = sum( *b );
   avgb /= (1.0*numPixels);

   Program AuxVariance = RM_BEGIN  // return an array holding the squares of x - x_mean
    {
      In<Value1ui>	x;
      In<Value1f>	avg;
      Out<Value1f>	var;

      Value1f temp = x-avg;
      var = temp*temp;
    }
   RM_END

   res = AuxVariance(*a, avga);
   covaa = sum( res );
//    covaa /= (1.0*numPixels);

   res = AuxVariance(*b, avgb);
   covbb = sum( res );
//    covbb /= (1.0*numPixels);

   Program AuxCovariance = RM_BEGIN // return an array holding the (x - x_mean)(y-y_mean) products
    {
      In<Value1ui>	x;
      In<Value1ui>	y;
      In<Value1f>	avgx;
      In<Value1f>	avgy;
      Out<Value1f>	var;

      var = (x-avgx)*(y-avgy);
    }
   RM_END

   res = AuxCovariance(*a, *b, avga, avgb);
   covab = sum( res );
//    covab /= (1.0*numPixels);

   aux = covab*covab / (covaa*covbb);
   return aux.get_value(0);
}
//***********************************************************
double getSpeed()
{
 const double CORE2SLOPE=0.20302;
 const double P4SLOPE=8.63539;
 const double CORE2FREQ=1795.517;
 const double P4FREQ= 2799.564;
 double slope;
 double cpufreq;
 int cpumodel;
 int cpufamily;
 char buff[100]="";
 #ifdef _LINUX_EXEC_
  ifstream fin("/proc/cpuinfo");
 #else
  ifstream fin("c:\\gpgpu.dat");
 #endif

/*
processor       : 0
vendor_id       : AuthenticAMD
cpu family      : 15
model           : 75
model name      : AMD Athlon(tm) 64 X2 Dual Core Processor 3800+
stepping        : 2
cpu MHz         : 2009.290*/

   while(strcmp(buff,"family")) fin >> buff;
   fin >> buff;
   fin >> cpufamily;

   while(strcmp(buff,"model")) fin >> buff;
   fin >> buff;
   fin >> cpumodel;

   while(strcmp(buff,"MHz")) fin >> buff;
   fin >> buff;
   fin >> cpufreq;
   fin.close();

   if(cpumodel==2 && cpufamily==15) // P4
      slope = P4SLOPE * P4FREQ/cpufreq;
   else
      slope = CORE2SLOPE;
 return slope/IMGSIZE;  // because load is expressed in bytes
}
//***********************************************************
// global data shared by the two threads
Image a, b; // a is the one to match and b the one from the pool. Initialized in main
unsigned int *apix;
unsigned int *bpix; // only one needed 
int imgin=0, imgout=0;
sem_t ready, space;
Array<2, Value1ui> *rpa; // for main image;
Array<2, Value1ui> *rpb[IMGBUFFSIZE]; // for images to compare with
Array<2, Value1ui> *rpc; // for result

   // arrays for probability calculations
Array<2, Value1f> *pij[IMGBUFFSIZE];
Array<1, Value1f> *pi; 
Array<1, Value1f> *pj[IMGBUFFSIZE];

//***********************************************************
// runs in a separate thread, taking care of I/O
void *processIO(void *param)
{
   char buff[100];
   int stImage = ((int *)param)[0];
   int endImage = ((int *)param)[1];
   int numImages = endImage-stImage+1;

// cout << "Thread started\n";
   for (int picNum = 0; picNum < numImages; picNum++)
      {
        sem_wait(&space);
// cout << picNum << endl;

        // read image the next image to compare with
        sprintf(buff,"(%i).pgm",picNum+stImage);
        readImage(b, buff);

        // and make the transfer to GPU memory
        unsigned int *bpix = rpb[imgin] -> write_data();
        for(int i=0;i<a.height;i++)
           for(int j=0;j<a.width;j++)
              bpix[i*a.width+j] = b.pixel.cp[i*a.width+j];
         CalcProb(pj[imgin], bpix, a.width*a.height, a.levels);
         CalcJointProb(pij[imgin], apix, bpix, a.width*a.height, a.levels);
         imgin = (imgin+1) % IMGBUFFSIZE;
         sem_post(&ready);
      }
   pthread_exit(NULL);
// cout << "Thread exited\n";
}
//***********************************************************
// called by each compute node. Returns the index of the best image as stored in the rr array
int process(int *limits)
{
   int stImage = limits[0], endImage= limits[1];
   int numImages = endImage-stImage+1;
   rr = new ImgRegResults[numImages]; // registration results
   pthread_t thrID;

   readImage(a, "main.pgm");


   // allocate all arrays
   rpa = new Array<2, Value1ui>(a.width, a.height);
   for(int i=0;i<IMGBUFFSIZE;i++)
      rpb[i] = new Array<2, Value1ui>(a.width, a.height);
   rpc = new Array<2, Value1ui>(a.width, a.height);

   // arrays for probability calculations
   pi = new Array<1, Value1f>(a.levels); 
   for(int i=0;i<IMGBUFFSIZE;i++)
     {
      pij[i] = new Array<2, Value1f>(a.levels, a.levels);
      pj[i]  = new Array<1, Value1f>(a.levels);
     }

   // arrays for difference calculations
   Array<2, Value1f> d1(a.levels, a.levels);
   float *diff1 = d1.write_data();
   for(int i=0;i<a.levels;i++)
     for(int j=0;j<a.levels;j++)
        diff1[i*a.levels + j] = abs(i-j);

   // prepare data arrays for image a
   apix = rpa->write_data();
   for(int i=0;i<a.height;i++)
     for(int j=0;j<a.width;j++)
       apix[i*a.width+j] = a.pixel.cp[i*a.width+j];
   CalcProb(pi, apix, a.width*a.height, a.levels);

   sem_init(&ready,0,0);
   sem_init(&space, 0, IMGBUFFSIZE);
   assert(pthread_create(&thrID, NULL, processIO, limits)==0);

   for (int picNum = 0; picNum < numImages; picNum++)
      {
        sem_wait(&ready);
// cout << "Processing " << picNum << endl;
        // Calculate Covariance
        rr[picNum].C = Covariance(rpa, rpb[imgout], a.width * a.height);

        Value2f diffres = Diff(pij[imgout], d1);
        rr[picNum].ID=picNum+stImage;
        rr[picNum].D1=diffres.get_value(0);
        rr[picNum].D2=diffres.get_value(1);
        // mutual information calculation has to be done after difference calculation 
        // because the pij array is modified
        rr[picNum].M =mutualInfo(pi, pj[imgout], pij[imgout], a.levels);
        //rr[picNum].M =mutual(apix, bpix, a.width*a.height, a.levels);

//          cout << rr[picNum].C << " " << rr[picNum].M << " " <<rr[picNum].D1 << " " <<rr[picNum].D2 <<endl;
        imgout = (imgout+1) % IMGBUFFSIZE;
        sem_post(&space);
      }
   
   // return the data for the best image (only mutual info used here)
   int bestidx=0;
   for(int i=1;i<numImages;i++)
    if(rr[bestidx].M < rr[i].M)
       bestidx = i;



   //wait for other thread to join in
   char *res;
// cout << "Attempting to end process " << bestidx << endl;
   pthread_join(thrID, (void **)&res);
// cout << "Processing ended " << bestidx << endl;
// cout << "Attempting to release memory " << bestidx << endl;

   // release all memory
   delete rpa;
   for(int i=0;i<IMGBUFFSIZE;i++)
      delete rpb[i];
   delete rpc;
   delete pi; 
   for(int i=0;i<IMGBUFFSIZE;i++)
     {
      delete pij[i];
      delete pj[i];
     }

   return bestidx;
}
//***********************************************************
double partition2(double *p, int N, int L, int b, int d, double l, double *part, int *quant)
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
  
  qsort(v, N, sizeof(nodeLoad), comp);

  for(int i=0;i<diff;i++)
    quant[v[i].ID] ++;
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
   MPI::Status status;

   MPI::Init(argc, argv);
   rapidmind::init();
//   use_backend("cc");
//   use_backend("glsl");

   // get essential info
   stTime = MPI::Wtime();
   myid = MPI::COMM_WORLD.Get_rank ();
   num_proc = MPI::COMM_WORLD.Get_size ();
   numImages = atoi(argv[1]);

   //collect node speed and run partitioning code
   nodeSpeed = getSpeed();
   if(myid==0)
    {
      allspeeds = new double[num_proc];
      limitsDist = new int[num_proc*2];
    }

// cout << myid << " " << nodeSpeed << endl;
//   MPI::COMM_WORLD.Gather( &nodeSpeed, 1, MPI::DOUBLE, allspeeds, num_proc, MPI::DOUBLE, 0);

   if(myid==0)
      {
        for(int i=1;i<num_proc;i++)
           MPI::COMM_WORLD.Recv( allspeeds+i, 1, MPI::DOUBLE, i, SPEEDTAG, status);

        allspeeds[0] = nodeSpeed;
// cout << allspeeds[0] << " " << allspeeds[1] << endl;
        double part[num_proc];
        int imgcount[num_proc];
        double disk_rate[]={4.76, 9.44/2, 10.9/3, 10.96/4}; // measured MB per second
        double p[]={1.0e-6,2.0e-6,3.0e-6}; // these are arbitrary. MHA has to fix
        double l = 1.0/ (1024*1024*disk_rate[num_proc]); // convert to sec/B units
        partition(allspeeds, num_proc, numImages * IMGSIZE,IMGSIZE,20,l, part, imgcount);

        cout << imgcount[0] << " " << imgcount[1] << endl;
        limitsDist[0] = 0;
        limitsDist[1] = imgcount[0]-1;
        for(int i=1;i<num_proc;i++)
          {
            limitsDist[2*i] = limitsDist[2*i-1]+1;
            limitsDist[2*i+1] = limitsDist[2*i] + imgcount[i]-1;
          }

for(int i=0;i< 2*num_proc; i++)
   cout << limitsDist[i] << " ";
cout << endl;

        limits[0] = limitsDist[0];  // just for node 0
        limits[1] = limitsDist[1];

        // send the limits to the rest
        for(int i=1;i<num_proc;i++)
           MPI::COMM_WORLD.Send( limitsDist+2*i, 2, MPI::INT, i, ASSTAG);
      }
    else
      {
        MPI::COMM_WORLD.Send( &nodeSpeed, 1, MPI::DOUBLE, 0, SPEEDTAG);
        MPI::COMM_WORLD.Recv( limits, 2, MPI::INT, 0, ASSTAG, status);
      }


//    MPI::COMM_WORLD.Scatter( limitsDist, 2*num_proc, MPI::INT, limits, 2, MPI::INT, 0);
   cout << "Node " << myid << " is assigned from " << limits[0] << " to " << limits[1] << endl;
   int bestidx = process(limits);

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

   rapidmind::finish();
   MPI::Finalize();
   return 0;
}
