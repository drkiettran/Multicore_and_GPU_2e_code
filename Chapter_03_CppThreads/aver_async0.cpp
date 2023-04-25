/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ aver_async0.cpp -o aver_async0 -pthread
 ============================================================================
 */

#include <iostream>
#include <future>
#include <numeric>
#include <vector>
#include <cmath>
#include <functional>

using namespace std;

double partialSum(vector<double>::iterator start, vector<double>::iterator end)
{
   double partialRes = 0;
   for(auto i=start; i<end; i++)
       partialRes += *i;

   return partialRes;
}
//------------------------------------------
int main(int argc, char **argv)
{
  int numThr=atoi(argv[1]);
  int N = atoi(argv[2]);
  int step=(int)ceil(N*1.0/numThr);  
  double res=0;
  
  vector<double> data(N);
  iota(data.begin(), data.end(), 1);
  
  future<double> f[numThr];
  vector<double>::iterator localStart = data.begin();
  vector<double>::iterator localEnd; 
  for(int i=0;i<numThr;i++)
  {
      localEnd=localStart+step;
      if(i==numThr-1) localEnd=data.end();
      f[i] = async(partialSum, localStart, localEnd);
      localStart += step;
  }
 
  for(int i=0;i<numThr;i++)
      res += f[i].get();    
  cout << "Average is : " << res/N << endl;
  return 0;  
}
