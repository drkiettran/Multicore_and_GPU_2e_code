/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : August 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ aver1.cpp -o aver1 -pthread
 ============================================================================
 */

#include <iostream>
#include <thread>
#include <numeric>
#include <vector>
#include <cmath>
#include <mutex>
#include <functional>

using namespace std;

void partialSum(vector<double>::iterator start, vector<double>::iterator end, double &res,  mutex &l)
{
   double partialRes = 0;
   for(auto i=start; i<end; i++)
       partialRes += *i;

    l.lock();
    res += partialRes;
    l.unlock();
}
//------------------------------------------
int main(int argc, char **argv)
{
  int numThr=atoi(argv[1]);
  int N = atoi(argv[2]);
  int step=(int)ceil(N*1.0/numThr);  
  double res=0;
  mutex l;
  
  vector<double> data(N);
  iota(data.begin(), data.end(), 1);
  thread *thr[numThr];
  vector<double>::iterator localStart = data.begin();
  vector<double>::iterator localEnd; 
  for(int i=0;i<numThr;i++)
  {
      localEnd=localStart+step;
      if(i==numThr-1) localEnd=data.end();
      thr[i] = new thread(partialSum, localStart, localEnd, ref(res), ref(l));
      localStart += step;
  }
 
  for(int i=0;i<numThr;i++)
  {
      thr[i]->join();
      delete thr[i];    
  }
  cout << "Average is : " << res/N << endl;
  return 0;  
}
