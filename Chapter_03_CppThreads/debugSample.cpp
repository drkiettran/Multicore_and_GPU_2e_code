/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake debugSample.pro; make
 ============================================================================
 */
#include <thread>
#include <mutex>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <time.h>
#include <chrono>

using namespace std;

#define DEBUG

//***********************************************
chrono::high_resolution_clock::time_point time0;
mutex l;
double hrclock ()
{
  chrono::high_resolution_clock::time_point t;
  t = chrono::high_resolution_clock::now();
  return chrono::duration<double>(t-time0).count(); //defaults to seconds
}

//***********************************************
void debugMsg (string msg, double timestamp)
{
  l.lock ();
  cerr << timestamp << " " << msg << endl;
  l.unlock ();
}

//***********************************************
int counter = 0;

class MyThread
{
private:
  int ID;
  int runs;
public:
  MyThread (int i, int r):ID (i), runs (r) {}
  void operator()()
  {
    cout << "Thread " << ID << " is running\n";
    for (int j = 0; j < runs; j++)
      {
#ifdef DEBUG
        ostringstream ss;
        ss << "Thread #" << ID << " counter=" << counter;
        debugMsg (ss.str (), hrclock ());
#endif
        this_thread::sleep_for(chrono::duration < int, milli > (rand () % 4 + 1));
        counter++;
      }
  }
};

int main (int argc, char *argv[])
{
#ifdef DEBUG
  time0  = chrono::high_resolution_clock::now(); 
#endif

  srand (time (0));
  int N = atoi (argv[1]);
  int runs = atoi (argv[2]);
  unique_ptr<thread> t[N];
  unique_ptr<MyThread> mt[N];
  for (int i = 0; i < N; i++)
  {
      mt[i] = make_unique<MyThread>(i, runs);
      t[i] = make_unique<thread>( ref(*mt[i]));
  }

  for (int i = 0; i < N; i++)
    t[i]->join ();

  cout << counter << endl;
  return 0;
}
