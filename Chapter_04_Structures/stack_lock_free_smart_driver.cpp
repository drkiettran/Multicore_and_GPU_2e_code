/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ -std=c++17 stack_lock_free_smart_driver.cpp -o stack_lock_free_smart_driver -pthread -latomic
 ============================================================================
 */

#include<iostream>
#include<thread>
#include<stdlib.h>
#include"stack_lock_free_smart.hpp"

using namespace std;

int main(int argc, char **argv)
{
  int N=atoi(argv[1]);
  unique_ptr<thread> t[N];
  stack_lock_free_smart<int> st;
  
  shared_ptr<int> aux;
  cout << atomic_is_lock_free(&aux) << endl;
  unique_ptr<int> aux2;
  cout << atomic_is_lock_free(&aux2) << endl;
  unsigned int seed=0;
  for(int i=0;i<N;i++)      
      t[i] = make_unique<thread>([&](){
          for(int i=0;i<10;i++)
          {
              int act=rand_r(&seed)%10;
              int x;
              if(act)
              {
                  x = rand_r(&seed)%100;
                  cout << x << " ";
                  st.push(x);
              }
              else
              {
                  st.pop(x);
              } 
          }
    });
  
  for(int i=0;i<N;i++)
      t[i]->join();
  cout << endl;
  int v;
  while(st.pop(v))
      cout << v << " ";
  cout << endl;
  return 0;
}
    
    
