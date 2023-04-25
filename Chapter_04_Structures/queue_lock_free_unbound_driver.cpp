/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ -std=c++17 queue_lock_free_unbound_driver.cpp -o queue_lock_free_unbound_driver -pthread -latomic
 ============================================================================
 */

#include<iostream>
#include<thread>
#include<stdlib.h>
#include"queue_lock_free_unbound_aba.hpp"

using namespace std;

// mutex l;
int main(int argc, char **argv)
{
  int N=atoi(argv[1]);
  unique_ptr<thread> t[N];
  queue_lock_free_unbound<int> q;
    int x=1;
  unsigned int seed=0;
  for(int i=0;i<N;i++)
      t[i] = make_unique<thread>([&](){
          for(int j=0;j<10;j++)
          {
              int act=rand_r(&seed)%4;
              int x = rand_r(&seed)%100;
              if(j<5)
              {
                 q.enque(x);
              }
              else
              {
                 x = q.deque();
              } 
          }
    });
  
  for(int i=0;i<N;i++)
      t[i]->join();
  
  cout << q.size() << endl;    
  q.dump_lists("FINAL");
  return 0;
}
    
    
