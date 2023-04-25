/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ -std=c++17 fiber_hello.cpp -o fiber_hello -lboost_fiber -lstdc++ -lboost_context
 ============================================================================
 */

#include <boost/fiber/all.hpp>
#include <iostream>
#include <chrono>
using namespace std;

void msg()
{
   boost::this_fiber::sleep_for(chrono::duration<int, milli>(rand()%100));
   cout <<"Hello from fiber " << boost::this_fiber::get_id() << endl;    
}

int main()
{
    boost::fibers::fiber f[10];
    for(int i=0;i<10;i++)
         f[i] = boost::fibers::fiber(msg);

    for(int i=0;i<10;i++)
         f[i].join();
}
