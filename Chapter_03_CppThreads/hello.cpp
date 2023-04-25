/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : June 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ hello.cpp -o hello -pthread
 ============================================================================
 */

#include <iostream>
#include <thread>

using namespace std;

void f()
{
  cout <<"Hello from thread " << this_thread::get_id() << "\n";
  this_thread::sleep_for(1s);
}

int main(int argc, char **argv)
{
  thread t(f);
  cout << "Main thread waiting...\n";
  t.join();
  return 0;  
}
