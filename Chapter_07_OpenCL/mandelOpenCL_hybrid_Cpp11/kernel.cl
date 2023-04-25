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
#include "const.h"

//************************************************************
int diverge (double cx, double cy)
{
  int iter = 0;
  double vx = cx, vy = cy, tx, ty;
  while (iter < MAXITER && (vx * vx + vy * vy) < 4)
    {
      tx = vx * vx - vy * vy + cx;
      ty = 2 * vx * vy + cy;
      vx = tx;
      vy = ty;
      iter++;
    }
  return iter;
}

//************************************************************
// kernel void mandelKernel (global unsigned char *d_res, double upperX, double upperY, double stepX, double stepY, int resX, int resY, int pitch)
kernel void mandelKernel (global unsigned char *d_res, double upperX, double upperY, double stepX, double stepY, int resX, int resY)
{
  int myX = get_global_id (0);
  int myY = get_global_id (1);
  int advX = get_global_size(0);
  int advY = get_global_size(1);

  int i, j;
  for (i = myX; i < resX; i+= advX)
    for (j = myY; j < resY; j+= advY)
      {
        double tempx, tempy;
        tempx = upperX + i * stepX;
        tempy = upperY - j * stepY;

        int color = diverge (tempx, tempy);
        d_res[j * resX + i] = color % 256;
      }
}

