/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ -std=c++17 fiber_pool.cpp -o fiber_pool -lboost_fiber -lboost_context -lboost_system -pthread
 ============================================================================
 */

#include <boost/fiber/all.hpp>
#include <thread>
#include <mutex>
#include <iostream>
#include <chrono>
#include "semaphore.cpp"

using namespace std;

//******************************************
thread workerSetup (boost::fibers::barrier & b, semaphore &done, int n)
{
  return move (thread ([&]()
                       {
                       boost::fibers::use_scheduling_algorithm < boost::fibers::algo::shared_work > ();
                       b.wait (); 
                       while(done.try_acquire()==false)
                           boost::this_fiber::yield();
                    }
               ));
}

//******************************************
void foo ()
{
  for(int i=0;i<10;i++)
  {
    boost::this_fiber::sleep_for (chrono::duration < int, milli > (rand () % 100));
    cout << "Hello #" << i << " from fiber " << boost::this_fiber::get_id () << " running on thread " << this_thread::get_id () << endl;
    boost::this_fiber::yield();
  }
}

//******************************************
int main ()
{
  int numThr = thread::hardware_concurrency ();
  boost::fibers::barrier b {static_cast < size_t > (numThr)};
  semaphore done (0);
  unique_ptr < thread > t[numThr - 1];

  boost::fibers::use_scheduling_algorithm < boost::fibers::algo::shared_work > ();

  for (int i = 0; i < numThr - 1; i++)
    t[i] = make_unique < thread > (workerSetup (b, done, numThr));

  // block until all threads set the fiber scheduler
  b.wait ();

  // create the fibers
  boost::fibers::fiber f[100];
  for (int i = 0; i < 100; i++)
    {
      f[i] = boost::fibers::fiber (foo);
    }

  // waiting for all work to complete
  for(int i=0;i<100;i++)
      f[i].join();

  // let threads know that they can stop.
  done.release (numThr-1);

  for (int i = 0; i < numThr - 1; i++)
    t[i]->join ();
  return 0;
}
