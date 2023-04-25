/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake writersFav.pro; make
 ============================================================================
 */
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <chrono>
#include <stdlib.h>

using namespace std;

const int NUMOPER = 3;
//*************************************
class Monitor
{
private:
  mutex l;
  condition_variable wq;        // for blocking writers
  condition_variable rq;        // for blocking readers
  int readersIn;                // how many readers in their critical section
  bool writerIn;                // set if a write is in its critical section
  int writersWaiting;           // how many writers are waiting to enter
public:
    Monitor ():readersIn (0), writerIn (0), writersWaiting (0) {}
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
  while (writerIn == true || writersWaiting > 0)
    rq.wait (ul);

  readersIn++;
}

//*************************************
void Monitor::canWrite ()
{
  unique_lock < mutex > ul (l);
  while (writerIn == true || readersIn > 0)
    {
      writersWaiting++;
      wq.wait (ul);
      writersWaiting--;
    }

  writerIn = true;
}

//*************************************
void Monitor::finishedReading ()
{
  lock_guard < mutex > lg (l);
  readersIn--;
  if (readersIn == 0)
    wq.notify_one ();
}

//*************************************
void Monitor::finishedWriting ()
{
  lock_guard < mutex > lg (l);
  writerIn = false;
  if (writersWaiting > 0)
    wq.notify_one ();
  else
    rq.notify_all ();
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
