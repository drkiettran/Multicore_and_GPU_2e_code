/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2018
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake terminProdConsMess.pro ; make
 ============================================================================
 */
#include <QThread>
#include <QSemaphore>
#include <QMutex>
#include <QTime>
#include <iostream>
#include <unistd.h>
#include <math.h>

using namespace std;

const int BUFFSIZE = 100;
const double LOWERLIMIT = 0;
const double UPPERLIMIT = 10;
//--------------------------------------------
typedef struct Slice
{
  double start;
  double end;
  int divisions;
} Slice;
//--------------------------------------------
double func (double x)
{
  return fabs (sin (x));
}

//--------------------------------------------
// acts as a consumer
class IntegrCalc:public QThread
{
private:
  int ID;
  static QSemaphore *slotsAvail;
  static QSemaphore *resAvail;
  static QMutex l2;
  static QMutex resLock;
  static Slice *buffer;
  static int out;
  static double *result;
  static QSemaphore numProducts;
public:
  static void initClass (QSemaphore * s, QSemaphore * a, Slice * b, double *r);

    IntegrCalc (int i):ID (i)
  {
  };
  void run ();
  double compute (double, double, int);
};

//---------------------------------------

QSemaphore *IntegrCalc::slotsAvail;
QSemaphore *IntegrCalc::resAvail;
QMutex IntegrCalc::l2;
QMutex IntegrCalc::resLock;
Slice *IntegrCalc::buffer;
int IntegrCalc::out = 0;
double *IntegrCalc::result;

//---------------------------------------
double IntegrCalc::compute (double st, double en, int div)
{
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
  return localRes;
}

//---------------------------------------
void IntegrCalc::initClass (QSemaphore * s, QSemaphore * a, Slice * b, double *res)
{
  slotsAvail = s;
  resAvail = a;
  buffer = b;
  result = res;
  *result = 0;
}

//---------------------------------------

void IntegrCalc::run ()
{
  while (1)
    {
      resAvail->acquire ();     // wait for an available item
      l2.lock ();
      int tmpOut = out;
      out = (out + 1) % BUFFSIZE;       // update the out index
      l2.unlock ();

      // take the item out
      double st = buffer[tmpOut].start;
      double en = buffer[tmpOut].end;
      double div = buffer[tmpOut].divisions;

      slotsAvail->release ();   // signal for a new empty slot 

      if (div == 0)
        break;                  // exit

      //calculate and accummulate partial area  
      double localRes = compute (st, en, div);
      resLock.lock ();
      *result += localRes;
      resLock.unlock ();
    }
}

//---------------------------------------

int main (int argc, char *argv[])
{
  QTime exect;
  exect.start ();

  if (argc == 1)
    {
      cerr << "Usage " << argv[0] << " #threads #trapezoids\n";
      exit (1);
    }
  int N = atoi (argv[1]);
  int trapezoids = atoi (argv[2]);
  int J = N *10;
  
  double result = 0;
  double divLen = (UPPERLIMIT - LOWERLIMIT) / trapezoids;
  double st, end = LOWERLIMIT;
  int divisionsPerJob = 1.0*trapezoids/J + 1;
  
  if (N == 1)                   // sequential execution
    {
      IntegrCalc c{0};
      result = c.compute (st, end, trapezoids);  // no extra threads here
    }
  else
    {
      Slice *buffer = new Slice[BUFFSIZE];
      QSemaphore avail, buffSlots (BUFFSIZE);
      int in = 0;
      IntegrCalc::initClass (&buffSlots, &avail, buffer, &result);

      N--; // one thread used as a "producer"
      IntegrCalc *t[N];
      for (int i = 0; i < N; i++)
        {
          t[i] = new IntegrCalc (i);
          t[i]->start ();
        }

      // main thread is responsible for handing out 'jobs'
      // It acts as the producer in this setup
      for (int i = 0; i < J; i++)
        {
          st = end;
          end += divLen;
          if (i == J - 1)
            end = UPPERLIMIT;

          buffSlots.acquire ();
          buffer[in].start = st;
          buffer[in].end = end;
          buffer[in].divisions = divisionsPerJob;
          in = (in + 1) % BUFFSIZE;
          avail.release ();
        }

      // put termination sentinels in buffer
      for (int i = 0; i < N; i++)
        {
          buffSlots.acquire ();
          buffer[in].divisions = 0;
          in = (in + 1) % BUFFSIZE;
          avail.release ();
        }

      // wait for all threads to finish
      for (int i = 0; i < N; i++)
        t[i]->wait ();
      delete[]buffer;
    }
  cout << exect.elapsed () << endl;
//     cout << "Result : " << result << " in " << exect.elapsed() << endl;
  return 0;
}
