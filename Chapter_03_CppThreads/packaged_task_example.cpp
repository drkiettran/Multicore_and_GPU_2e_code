/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ -std=c++17 packaged_task_example.cpp -pthread -o packaged_task_example
 ============================================================================
 */

#include <future>
#include <iostream>
#include <thread>
#include <functional>
using namespace std;

double maxf (double a, double b)
{
  return (a > b) ? a : b;
}


void explicitRun2 ()
{
  packaged_task < double (double, double) > pt (maxf);
  future < double >res = pt.get_future ();
  pt (1, 2);
  cout << res.get () << endl;
}

void explicitRun ()
{
  packaged_task < double () > pt (bind(maxf,1,2));
  future < double >res = pt.get_future ();
  pt ();
  cout << res.get () << endl;
}

void threadRun ()
{
  packaged_task < double (double, double) > pt (maxf);
  future < double >res = pt.get_future ();
  thread t (move (pt), 1, 2);
  t.join ();
  cout << res.get () << endl;
}

int main ()
{
  explicitRun ();
  threadRun ();
}
