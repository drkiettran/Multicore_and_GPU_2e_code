/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ acquire-release.cpp -o acquire-release -pthread -latomic -std=c++17 -O2
 ============================================================================
 */
#include <thread>
#include <atomic>
#include <cassert>

using namespace std;

atomic<bool> x, y;

void thr1()
{
  x.store(true, memory_order_release);
  y.store(true, memory_order_release);
}

void thr2()
{
  bool temp=false;
  while(!temp)
      temp=y.load(memory_order_acquire);
  assert(x.load(memory_order_acquire) == true);
}

//------------------------------------------
int main (int argc, char **argv)
{
    thread t1(thr1);
    thread t2(thr2);
    t1.join();
    t2.join();
    return 0;
}
