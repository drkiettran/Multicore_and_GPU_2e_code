/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : June 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ aver.cpp -o aver -pthread
 ============================================================================
 */

#include <iostream>
#include <thread>
#include <numeric>
#include <vector>
#include <cmath>

using namespace std;

void partialSum(vector<double>::iterator start, vector<double>::iterator end, double *res)
{
    *res=0;
    for(auto i=start; i<end; i++)
        *res += *i;
}
//------------------------------------------
int main(int argc, char **argv)
{
  int numThr=atoi(argv[1]);
  int N = atoi(argv[2]);
  int step=(int)ceil(N*1.0/numThr);  
  double res[numThr];
  
  vector<double> data(N);
  iota(data.begin(), data.end(), 1);
  thread *thr[numThr];
  vector<double>::iterator localStart = data.begin();
  vector<double>::iterator localEnd; 
  for(int i=0;i<numThr;i++)
  {
      localEnd=localStart+step;
      if(i==numThr-1) localEnd=data.end();
      thr[i] = new thread(partialSum, localStart, localEnd, res+i);
      localStart += step;
  }
 
  double total=0;
  for(int i=0;i<numThr;i++)
  {
      thr[i]->join();
      delete thr[i];    
      total += res[i];
  }
  cout << "Average is : " << total/N << endl;
  return 0;  
}
