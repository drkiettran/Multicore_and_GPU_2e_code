/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ atomic_lock.cpp -o atomic_lock -pthread -latomic -std=c++17 -O2
 ============================================================================
 */

#include <thread>
#include <atomic>
#include <memory>
#include <iostream>

using namespace std;

//------------------------------------------
class ALock
{
private:
  atomic < bool >lockState;

public:
  void lock();
  void unlock ();
  ALock ()
  {
    lockState.store (false);
  }
  ALock (ALock &) = delete;
//   ALock &operator=()(const ALock &o) = delete;
};

//------------------------------------------
void ALock::lock()
{
  while (true)
    {
      bool temp=false;
      if (lockState.compare_exchange_weak (temp, true, memory_order_acq_rel ))
          return;
      this_thread::yield ();
    }
}

//------------------------------------------
void ALock::unlock ()
{
    lockState.store(false, memory_order_release);
}
//------------------------------------------

ALock l;
int count=0;
//------------------------------------------

void thr1()
{
  for(int i=0;i<100;i++)
  {
      l.lock();
      count++;
      l.unlock();
  }    
}


//------------------------------------------
int main (int argc, char **argv)
{
  int N = atoi (argv[1]);
  
  unique_ptr < thread > t[N];
  for (int i = 0; i < N; i++)
    {
      t[i] = make_unique < thread > (thr1);
    }

  for (int i = 0; i <N; i++)
    t[i]->join ();
  cout << count << endl;
  return 0;
}
