/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ -std=c++17 list_lock_free_driver.cpp -o list_lock_free_driver -pthread -latomic
 ============================================================================
 */

#include<iostream>
#include<thread>
#include<time.h>
#include<stdlib.h>
// #include"list_lock_free.cpp"
// #include"list_lock_free_v2.cpp"
#include"list_lock_free_bounded.hpp"
// #include"list_lock_free_hazard.cpp"

using namespace std;

const int RNG=10;
int main(int argc, char **argv)
{
  int N=atoi(argv[1]);
  unique_ptr<thread> t[N];
  list_lock_free_bounded<int> lst(RNG);
  srand(time(NULL));

  unsigned int seed=0;
  for(int i=0;i<N;i++)
      t[i] = make_unique<thread>([&](){
          for(int j=0;j<10;j++)
          {
              int x = rand_r(&seed)%RNG;
              if(j<5)
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
  lst.dump_lists("Final");
  return 0;
}
    
    
