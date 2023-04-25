/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : November 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ -std=c++17 queue_performance_test.cpp -o queue_performance_test -pthread -latomic
 ============================================================================
 */

#include<iostream>
#include<thread>
#include<stdlib.h>
#include"queue_lock_free_unbound_aba.hpp"
// #include"queue_lock_free_bounded_aba.hpp"
#include"queue_fine.hpp"
#include <chrono>


using namespace std;
using namespace std::chrono;

int main (int argc, char **argv)
{
  int Niter = atoi (argv[1]);
  int numThr = atoi (argv[2]);
  int numOper = atoi (argv[3]);
  unique_ptr < thread > t[numThr];

  auto t1 = high_resolution_clock::now ();
  unsigned int seed = 0;
  queue < int >q_fine;
  for (int iter = 0; iter < Niter; iter++)
    {
      for (int i = 0; i < numThr; i++)
        t[i] = make_unique < thread > ([&]()
                                       {
                                       for (int j = 0; j < numOper; j++)
                                       {
                                       int x ;
                                       if (j <= numOper / 2) 
                                       {
                                           x = rand_r (&seed) % 100; 
                                           q_fine.push_back (x);
                                       }
                                       else
                                           x = q_fine.pop_front ();
                                           
                                       }
                                       }
      );
      for (int i = 0; i < numThr; i++)
        t[i]->join ();
    }
  auto t2 = high_resolution_clock::now ();
  cout << "Fine : " << duration_cast < milliseconds > (t2 - t1).count ();

  t1 = high_resolution_clock::now ();
  for (int iter = 0; iter < Niter; iter++)
    {
      queue_lock_free_unbound < int >qlfu;
      for (int i = 0; i < numThr; i++)
        t[i] = make_unique < thread > ([&]()
                                       {
                                       for (int j = 0; j < numOper; j++)
                                       {
                                       int x;  
                                       if (j <= numOper / 2)
                                       {
                                           x = rand_r (&seed) % 100;
                                           qlfu.enque (x);
                                       }
                                       else
                                           x = qlfu.deque ();
                                       }
                                       }
      );
        
      for (int i = 0; i < numThr; i++)
        t[i]->join ();
    }
   t2 = high_resolution_clock::now ();
   cout << " LF unbounded : " << duration_cast < milliseconds > (t2 - t1).count () << endl;
   return 0;
}
