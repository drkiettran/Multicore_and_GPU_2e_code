/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : July 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ aver2.cpp -o aver2 -pthread -std=c++14
 ============================================================================
 */

#include <iostream>
#include <thread>
#include <numeric>
#include <vector>
#include <cmath>
#include <functional>
#include <memory>

using namespace std;

struct PartialSumFunctor
{
  vector < double >::iterator start;
  vector < double >::iterator end;
  double res = 0;

  void operator () ();
};

void PartialSumFunctor::operator () ()
{
  for (auto i = start; i < end; i++)
    res += *i;
}
//------------------------------------------
int main (int argc, char **argv)
{
  int numThr = atoi (argv[1]);
  int N = atoi (argv[2]);
  int step = (int) ceil (N * 1.0 / numThr);

  vector < double >data (N);
  iota (data.begin (), data.end (), 1);
  vector < double >::iterator localStart = data.begin ();
  vector < double >::iterator localEnd;
  unique_ptr < thread > thr[numThr - 1];
  PartialSumFunctor f[numThr];
  for (int i = 0; i < numThr - 1; i++)
    {
      localEnd = localStart + step;
      f[i].start = localStart;
      f[i].end = localEnd;
      thr[i] = make_unique < thread > (ref (f[i]));
      localStart += step;
    }
  f[numThr - 1].start = localStart;
  f[numThr - 1].end = data.end ();
  f[numThr - 1] ();
  double total = f[numThr - 1].res;

  for (int i = 0; i < numThr - 1; i++)
    {
      thr[i]->join ();
      total += f[i].res;
    }
  cout << "Average is : " << total / N << endl;
  return 0;
}
