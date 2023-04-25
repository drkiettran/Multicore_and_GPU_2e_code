/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : July 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ aver3.cpp -o aver3 -pthread -lm -std=c++14
 ============================================================================
 */

#include <iostream>
#include <thread>
#include <numeric>
#include <vector>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>

using namespace std;

struct PartialSumFunctor
{
  vector < double >::iterator start;
  vector < double >::iterator end;
  double res=0;
   
  void operator () ();
};

void PartialSumFunctor::operator () ()
{
  for (auto i = start; i < end; i++)
    res += *i;
}

//------------------------------------------
class ThreadGuard
{
private:
    thread &thr;
public:
    explicit ThreadGuard(thread &t) : thr(t) {}
    ~ThreadGuard()
    {
        if(thr.joinable())
           thr.join();
    }

   ThreadGuard(const ThreadGuard &o) = delete;
   ThreadGuard & operator=()(const ThreadGuard &o) = delete;
};
//------------------------------------------
void process_aux(vector<double> &data, int numThr, vector<PartialSumFunctor> &f)
{
   int N=data.size();
   int step = (int) ceil (N * 1.0 / numThr);

   unique_ptr < thread > thr[numThr];
   unique_ptr < ThreadGuard > tg[numThr];

   vector < double >::iterator localStart = data.begin ();
   vector < double >::iterator localEnd;
   for (int i = 0; i < numThr; i++)
    {
      localEnd = localStart + step;
      if(i==numThr-1)
          localEnd = data.end();
      f[i].start = localStart;
      f[i].end = localEnd;
      thr[i] = make_unique<thread>(ref(f[i]));
      tg[i] =  make_unique<ThreadGuard>(ref(*thr[i]));
      localStart += step;
   }
}
//------------------------------------------
double process(vector<double> &data)
{
   int numThr = thread::hardware_concurrency();
   vector<PartialSumFunctor> f;
   f.resize(numThr);
   process_aux(data, numThr, f);
   
   double total=0;
   for (int i = 0; i < numThr; i++)
     total+= f[i].res;
   
   return total/data.size();
}
//------------------------------------------
int main (int argc, char **argv)
{
  int N = atoi (argv[1]);

  vector < double >data (N);
  iota (data.begin (), data.end (), 1);
  double aver = process(data);
  
  cout << "Average is : " << aver << endl;
  return 0;
}
