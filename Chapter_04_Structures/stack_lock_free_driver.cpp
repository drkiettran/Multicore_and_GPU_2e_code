/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ -std=c++17 stack_lock_free_driver.cpp -o stack_lock_free_driver -pthread -latomic
 ============================================================================
 */

#include<iostream>
#include<thread>
#include<stdlib.h>
// #include"stack_lock_free.hpp"
#include"stack_lock_free_v2.hpp"

using namespace std;

// mutex l;
int main(int argc, char **argv)
{
  int N=atoi(argv[1]);
  unique_ptr<thread> t[N];
  stack_lock_free<int> st;
  unsigned int seed=0;
  for(int i=0;i<N;i++)      
      t[i] = make_unique<thread>([&](){
          for(int i=0;i<10;i++)
          {
              int act=rand_r(&seed)%10;
              if(act)
              {
                  int x = rand_r(&seed)%100;
                  cout << x << " ";
                  st.push(x);
              }
              else
              {
                  int x = st.pop();
              } 
          }
    });
  
  for(int i=0;i<N;i++)
      t[i]->join();
  
      try
      {
  while(true)
      cout << st.pop() << "\n"; // Will terminate with an exception
      }
      catch(exception &x)
      {
          cout << endl;
      }
  return 0;
}
    
    
