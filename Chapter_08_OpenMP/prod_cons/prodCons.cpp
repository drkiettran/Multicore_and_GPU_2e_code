/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.1
 Last modified : March 2020
 License       : Released under the GNU GPL 3.0
 Description   : 1 producer, multiple consumers solution, combining OpenMP and C++11
 To build use  : qmake; make
 ============================================================================
 */
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mutex>
#include <vector>
#include <memory>
#include "semaphore.h"

using namespace std;

const int BUFFSIZE = 10;
const double LOWERLIMIT = 0;
const double UPPERLIMIT = 10;

const int NUMCONSUMERS = 2;
//--------------------------------------------
struct Slice
{
  double start;
  double end;
  int divisions;
};
//--------------------------------------------
double func (double x)
{
  return fabs (sin (x));
}

//--------------------------------------------
void integrCalc (shared_ptr < vector < Slice > >buffer, semaphore & buffSlots, semaphore & avail, mutex & l, int &out, mutex & resLock, double &res)
{
  while (1)
    {
      avail.acquire ();         // wait for an available item
      l.lock ();
      // take the item out
      double st = buffer->at (out).start;
      double en = buffer->at (out).end;
      double div = buffer->at (out).divisions;
      out = (out + 1) % BUFFSIZE;       // update the out index
      l.unlock ();

      buffSlots.release ();     // signal for a new empty slot 

      if (div == 0)
        break;                  // exit

      //calculate area  
      double localRes = 0;
      double step = (en - st) / div;
      double x;
      x = st;
      localRes = func (st) + func (en);
      localRes /= 2;
      for (int i = 1; i < div; i++)
        {
          x += step;
          localRes += func (x);
        }
      localRes *= step;

      // add it to result
      resLock.lock ();
      res += localRes;
      resLock.unlock ();
    }
}

//--------------------------------------------
int main (int argc, char **argv)
{
  if (argc == 1)
    {
      cerr << "Usage " << argv[0] << " #jobs\n";
      exit (1);
    }
  int J = atoi (argv[1]);
  shared_ptr < vector < Slice > >buffer = make_shared < vector < Slice >> (BUFFSIZE);
  int in = 0, out = 0;
  semaphore avail, buffSlots (BUFFSIZE);
  mutex l, integLock;
  double integral = 0;
#pragma omp parallel sections default(none) shared(buffer, in, out, avail, buffSlots, l, integLock, integral, J)
  {
// producer part    
#pragma omp section
    {
      // producer thread, responsible for handing out 'jobs'
      double divLen = (UPPERLIMIT - LOWERLIMIT) / J;
      double st, end = LOWERLIMIT;
      for (int i = 0; i < J; i++)
        {
          st = end;
          end += divLen;
          if (i == J - 1)
            end = UPPERLIMIT;

          buffSlots.acquire ();
          buffer->at (in).start = st;
          buffer->at (in).end = end;
          buffer->at (in).divisions = 1000;
          in = (in + 1) % BUFFSIZE;
          avail.release ();
        }

      // put termination sentinels in buffer
      for (int i = 0; i < NUMCONSUMERS; i++)
        {
          buffSlots.acquire ();
          buffer->at (in).divisions = 0;
          in = (in + 1) % BUFFSIZE;
          avail.release ();
        }
    }

// 1st consumer part
#pragma omp section
    {
      integrCalc (buffer, buffSlots, avail, l, out, integLock, integral);
    }


// 2nd consumer part
#pragma omp section
    {
      integrCalc (buffer, buffSlots, avail, l, out, integLock, integral);
    }
  }

  cout << "Result is : " << integral << endl;

  return 0;
}
