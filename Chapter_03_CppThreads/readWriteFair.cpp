/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake readWriteFair.pro; make
 ============================================================================
 */
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <chrono>
#include <stdlib.h>

using namespace std;

const int QUESIZE = 100;
const int NUMOPER = 3;
//*************************************

class Monitor
{
private:
  mutex l;
  condition_variable c[QUESIZE];        // a different condition for each waiting thread
  bool writeflag[QUESIZE];      // what kind of threads wait?
  condition_variable quefull;   // used when queue of waiting threads becomes full
  int in, out, counter;
  int readersIn;                // how many readers in their critical section
  int writersIn;                // how many writers in their critical section (0 or 1)
public:

  Monitor ():in (0), out (0), counter (0), readersIn (0), writersIn (0) {}
  void canRead ();
  void finishedReading ();
  void canWrite ();
  void finishedWriting ();
};

//*************************************

class Reader
{
private:
  int ID;
  Monitor *coord;
public:

  Reader (int i, Monitor * c):ID (i), coord (c) {}
  void operator () ();
};

//*************************************

void Reader::operator () ()
{
  for (int i = 0; i < NUMOPER; i++)
    {
      coord->canRead ();
      cout << "Reader " << ID << " read oper. #" << i << endl;
      this_thread::sleep_for (chrono::duration < int, milli > (rand () % 4 + 1));
      coord->finishedReading ();
    }
}

//*************************************

class Writer
{
private:
  int ID;
  Monitor *coord;
  int delay;
public:

  Writer (int i, Monitor * c):ID (i), coord (c) {}
  void operator () ();
};

//*************************************

void Writer::operator () ()
{
  for (int i = 0; i < NUMOPER; i++)
    {
      coord->canWrite ();
      cout << "Writer " << ID << " write oper. #" << i << endl;
      this_thread::sleep_for (chrono::duration < int, milli > (rand () % 4 + 1));
      coord->finishedWriting ();
    }
}

//*************************************

void Monitor::canRead ()
{
  unique_lock < mutex > ul (l);
  while (counter == QUESIZE)
    quefull.wait (ul);

  if (counter > 0 || writersIn)
    {
      int temp = in;
      writeflag[in] = false;
      in = (in + 1) % QUESIZE;
      counter++;
      c[temp].wait (ul);
    }
  readersIn++;
}

//*************************************

void Monitor::canWrite ()
{
  unique_lock < mutex > ul (l);
  while (counter == QUESIZE)
    quefull.wait (ul);

  if (counter > 0 || writersIn > 0 || readersIn > 0)
    {
      int temp = in;
      writeflag[in] = true;
      in = (in + 1) % QUESIZE;
      counter++;
      c[temp].wait (ul);
    }
  writersIn++;
}

//*************************************

void Monitor::finishedReading ()
{
  lock_guard < mutex > lg (l);
  readersIn--;
  if (readersIn == 0 && counter > 0)
    {
      c[out].notify_one ();     // it must be a writer that is being woken up
      out = (out + 1) % QUESIZE;
      counter--;
      quefull.notify_one ();
    }
}

//*************************************

void Monitor::finishedWriting ()
{
  lock_guard < mutex > lg (l);
  writersIn--;
  if (counter > 0)
    {
      if (!writeflag[out])
        {
          while (counter > 0 && !writeflag[out])        // start next readers
            {
              c[out].notify_one ();
              out = (out + 1) % QUESIZE;
              counter--;
            }
        }
      else                      // next writer
        {
          c[out].notify_one ();
          out = (out + 1) % QUESIZE;
          counter--;
        }
      quefull.notify_all ();
    }
}

//*************************************

int main (int argc, char **argv)
{
  if (argc == 1)
    {
      cerr << "Usage " << argv[0] << " #readers #writers\n";
      exit (1);
    }

  int numRead = atoi (argv[1]);
  int numWrite = atoi (argv[2]);
  Monitor m;
  shared_ptr < Reader > r[numRead];
  shared_ptr < Writer > w[numWrite];
  unique_ptr < thread > t[numRead + numWrite];

  srand (clock ());

  for (int i = 0; i < numRead; i++)
    {
      r[i] = make_shared < Reader > (i, &m);
      t[i] = make_unique < thread > (ref (*r[i]));
    }
  for (int i = 0; i < numWrite; i++)
    {
      w[i] = make_shared < Writer > (i, &m);
      t[numRead + i] = make_unique < thread > (ref (*w[i]));
    }

  for (int i = 0; i < numRead + numWrite; i++)
    t[i]->join ();

  return 0;
}
