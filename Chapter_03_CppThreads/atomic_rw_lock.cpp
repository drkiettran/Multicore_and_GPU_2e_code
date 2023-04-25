/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : August 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ atomic_rw_lock.cpp -o atomic_rw_lock.cpp -pthread -latomic -std=c++17
 ============================================================================
 */

#include <thread>
#include <atomic>
#include <memory>
#include <cstdio>

using namespace std;

const int UNLOCKED = 0;
const int WRITERIN = -1;
//------------------------------------------
class RWLock
{
private:
  atomic < int >lockState;

public:
  void lockRead ();
  void lockWrite ();
  void unlock ();
    RWLock ()
  {
    lockState.store (UNLOCKED);
  }
  RWLock (RWLock &) = delete;
//     RWLock &operator=()(RWLock &) = delete;
};

//------------------------------------------
void RWLock::lockRead ()
{
  while (true)
    {
      int currState = lockState.load ();
      if (currState >= 0)
        if (lockState.compare_exchange_weak (currState, currState + 1))
          return;
      this_thread::yield ();
    }
}

//------------------------------------------
void RWLock::lockWrite ()
{
  while (true)
    {
      int currState = lockState.load ();
//         printf("W %i\n", currState);
      if (currState == UNLOCKED)
        if (lockState.compare_exchange_weak (currState, WRITERIN))
          return;
      this_thread::yield ();
    }
}

//------------------------------------------
void RWLock::unlock ()
{
  while (true)
    {
      int currState = lockState.load ();
      if (currState == WRITERIN)
        {
          if (lockState.compare_exchange_weak (currState, UNLOCKED))
            return;
        }
      else
        {
          if (lockState.compare_exchange_weak (currState, currState - 1))
            return;
        }
    }
}

//------------------------------------------


struct Reader
{
  Reader (int i, shared_ptr < RWLock > &lr):ID (i), l (lr) {}
  void operator () ();
  int ID;
  shared_ptr < RWLock > l;
};
//------------------------------------------
void Reader::operator () ()
{
  l->lockRead ();
  // reader critical section
  printf ("Reader %i runs critical section\n", ID);
  l->unlock ();
}

//------------------------------------------
struct Writer
{
  Writer (int i, shared_ptr < RWLock > &lr):ID (i), l (lr) {}
  void operator () ();
  int ID;
  shared_ptr < RWLock > l;
};

//------------------------------------------
void Writer::operator () ()
{
  l->lockWrite ();
  // writer critical section
  printf ("Writer %i runs critical section\n", ID);

  l->unlock ();
}

//------------------------------------------
int main (int argc, char **argv)
{
  int W = atoi (argv[1]);
  int R = atoi (argv[2]);
  shared_ptr < RWLock > lp = make_shared < RWLock > ();

  shared_ptr < Reader > r[R];
  shared_ptr < Writer > w[W];
  unique_ptr < thread > t[W + R];
  for (int i = 0; i < W; i++)
    {
      w[i] = make_shared < Writer > (i, lp);
      t[i] = make_unique < thread > (ref (*w[i]));
    }
  for (int i = 0; i < R; i++)
    {
      r[i] = make_shared < Reader > (i, lp);
      t[i + W] = make_unique < thread > (ref (*r[i]));
    }

  for (int i = 0; i < R + W; i++)
    t[i]->join ();
  return 0;
}
