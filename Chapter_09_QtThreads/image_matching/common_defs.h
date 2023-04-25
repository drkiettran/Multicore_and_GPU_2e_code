#ifndef COMMONDEFS
#define COMMONDEFS

#define CPUHZ 2799564.0         /* 2009268.0 */
#define LLINT long long int

//#define MAXPIXELMEM 4096*4096
#define MAXPIXELMEM 1024*1024
#define MAXLEVELS 256
#define MAXTHREADS 128
#define WARP_BITS 5
#define WARP_N 10
#define THREAD_N (WARP_N << WARP_BITS)
#define BLOCK_MEMORY (WARP_N * MAXLEVELS)
#define MAX_GRIDBLOCKS  16
#define IMUL(a,b) __mul24(a,b)
#define IMGSIZE 786448


// MPI Comm tags
#define SPEEDTAG 0
#define ASSTAG 1
#define RESTAG 2

//***********************************************************
struct ImgRegResults
{
  int ID;
  float D1;
  float D2;
  float C;
  float M;
  float avg;  // allocated alongside the others to simplify CUDA memory allocation
  float x2;
};
//***********************************************************
struct Image
{
  int width;
  int height;
  int levels;
  unsigned char *pixel;
};
//***********************************************************
struct nodeLoad
{
  int ID;
  double left;
};
//**********************************************************
int process(int stImage, int endImage);

#endif