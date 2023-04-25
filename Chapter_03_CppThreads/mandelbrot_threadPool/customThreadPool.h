/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake ; make
 ============================================================================
 */
#include <thread>
#include <condition_variable>
#include <mutex>
#include <future>
#include <iostream>
#include <atomic>

using namespace std;

const int __BUFFERSIZE = 100;
const int __NUMTHREADS = 16;

template < typename T > class CustomThreadPool;
//************************************************************
template < typename T > class CustomThread
{
public:
  static CustomThreadPool < T > *tpool;
  void operator () ();
};

//************************************************************
template < typename T >
class CustomThreadPool
{
private:
  condition_variable empty;     // for blocking pool threads if buffer is empty
  condition_variable full;      // for blocking calling thread if buffer is empty
  mutex l;
  atomic < bool > done;         // flag for termination 
  unique_ptr < packaged_task < T () >> *buffer; // pointer to array of pointers to objects
  int in = 0, out = 0, count = 0, N, maxThreads;

  unique_ptr < thread > *t;     // threads RAII
public:
  CustomThreadPool (int nt = __NUMTHREADS, int n = __BUFFERSIZE);
  ~CustomThreadPool ();

  bool get (unique_ptr < packaged_task < T () >> &);    // to be called by the pool threads

  future < T > schedule (unique_ptr < packaged_task < T () > >);        // to be called for work request
};

//--------------------------------------
template < typename T >
CustomThreadPool < T > *CustomThread < T >::tpool;
//--------------------------------------
template < typename T >
void CustomThread < T >::operator () ()
{
  unique_ptr < packaged_task < T () >> tptr;
//   int count = 0;
  while (tpool->get (tptr))
    {
      packaged_task < T () > task = move (*tptr);
      task ();
//       count++;
    }
//   cout << this_thread::get_id () << " " << count << endl;
}

//************************************************************
template < typename T >
CustomThreadPool < T >::CustomThreadPool (int numThr, int n)
{
  N = n;
  done.store (false);
  buffer = new unique_ptr < packaged_task < T () >>[n]; // buffer init.
  maxThreads = numThr;
  t = new unique_ptr < thread >[maxThreads];    // thread pointers array alloc.
  CustomThread < T >::tpool = this;
  for (int i = 0; i < maxThreads; i++)  // starting pool threads
    t[i] = make_unique < thread > (CustomThread < T > ());
}

//--------------------------------------
template < typename T > 
CustomThreadPool < T >::~CustomThreadPool ()
{
  done.store (true);            // raise termination flag
  empty.notify_all ();          // wake up all pool threads that wait

  for (int i = 0; i < maxThreads; i++)
    this->t[i]->join ();

  delete[]t;
  delete[]buffer;
}

//--------------------------------------
template < typename T > 
future < T > CustomThreadPool < T >::schedule (unique_ptr < packaged_task < T () > >ct)
{
  unique_lock < mutex > ul (l);
  while (count == N)
    full.wait (ul);
  buffer[in] = move (ct);
  future < T > temp = buffer[in]->get_future ();
  in = (in + 1) % N;
  count++;

  empty.notify_one ();

  return move (temp);
}

//--------------------------------------
template < typename T > 
bool CustomThreadPool < T >::get (unique_ptr < packaged_task < T () >> &taskptr)
{
  unique_lock < mutex > ul (l);
  while (count == 0 && (done != true))
    empty.wait (ul);

  taskptr = move (buffer[out]);
  out = (out + 1) % N;
  count--;

  full.notify_one ();
  if (done.load () == true && count < 0)        // thread should call get again until there are no more pending tasks
    return false;
  else
    return true;
}
