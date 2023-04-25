/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : July 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ accountmonitor.cpp -o accountmonitor -pthread -std=c++17
 ============================================================================
 */
#include <iostream>
#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>

using namespace std;

class AccountMonitor
{
  private:
    condition_variable insufficientFunds;
    mutex m;
    double balance=0;
  public:
    void withdraw(double s) {
      unique_lock<mutex> ml(m);
      while(balance < s)    
          insufficientFunds.wait(ml);
      balance -= s;
     }

    void deposit(double s) {
      lock_guard<mutex> ml(m);
      balance += s;      
      insufficientFunds.notify_all();      
     }
};

int main()
{
   AccountMonitor ac; 
   thread t1([&](){ac.withdraw(100);});
   thread t2([&](){ac.deposit(200);});
   
   t1.join();
   t2.join();
   return 0;   
}
