/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ -std=c++17 list_lazy_driver.cpp -o list_lazy_driver -pthread -latomic
 ============================================================================
 */

#include<iostream>
#include<thread>
#include<stdlib.h>
#include"list_lazy.hpp"

using namespace std;

// mutex l;
int main(int argc, char **argv)
{
 int range=atoi(argv[1]);
  int ops=atoi(argv[2]);
  int N = thread::hardware_concurrency();
  unique_ptr<thread> t[N];
  list_lazy<int> lst;
  unsigned int seed=0;
  for(int i=0;i<N;i++)
      t[i] = make_unique<thread>([&](){
          for(int i=0;i<ops;i++)
          {
              int act=rand_r(&seed)%3;
              int x = rand_r(&seed)%range;
              if(act)
              {
                 lst.insert(x);
              }
              else
              {
                 lst.erase(x);
              } 
          }
    });
  
  for(int i=0;i<N;i++)
      t[i]->join();
  cout << lst.size() << endl;    
  return 0;
}
    
    
