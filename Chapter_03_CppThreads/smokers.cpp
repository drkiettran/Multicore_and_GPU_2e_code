/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake smokers.pro; make
 ============================================================================
 */
#include <iostream>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

using namespace std;

#define TOBACCO_PAPER 0  
#define TOBACCO_MATCHES 1
#define PAPER_MATCHES 2
#define MAXSLEEP 1000

const char *msg[]={"having matches", "having paper", "having tobacco"};
//***************************************************
class Monitor
{
  private:
     mutex l;
     condition_variable w, finish;
     int newingred;   
     int exitf;
  public:
     Monitor();
    // return 0 if OK. Otherwise it means termination
     int canSmoke(int);
     void newIngredient(int );
     void finishedSmoking();
     void finishSim();
};
//--------------------------------------------
Monitor::Monitor(): newingred(-1), exitf(0) {}
//--------------------------------------------
void Monitor::newIngredient(int newi)
{
  unique_lock< mutex > ul(l);  
  newingred = newi;
  w.notify_all();
  finish.wait(ul); // wait for smoker to finish
}
//--------------------------------------------
int Monitor::canSmoke(int missing)
{
  unique_lock< mutex > ul(l);  
  while(newingred != missing && ! exitf)
    w.wait(ul);
  return exitf;
}
//--------------------------------------------
void Monitor::finishedSmoking()
{
  lock_guard< mutex > lg(l);  
  newingred = -1;
  finish.notify_one();
}
//--------------------------------------------
void Monitor::finishSim()
{
  lock_guard< mutex > lg(l);      
  exitf=1;
  w.notify_all();
}
//***************************************************
class Smoker
{
  private:
    int missing_ingred;
    Monitor *m;
    int total;
  public:
    Smoker(int, Monitor *);
    void operator()();
};
//--------------------------------------------
Smoker::Smoker(int missing, Monitor *mon) : missing_ingred(missing), m(mon), total(0){}
//--------------------------------------------
void Smoker::operator()()
{
  while((m->canSmoke(missing_ingred)) ==0)
  {
     total++;
     cout << "Smoker " << msg[missing_ingred] << " is smoking\n";
     this_thread::sleep_for(chrono::duration<int, milli>(rand() % MAXSLEEP));
     m->finishedSmoking();
  }
// cout << "Smoker " << msg[missing_ingred] << " smoked a total of " << total << "\n";
}
//***************************************************
class Agent
{
  private:
    int runs;
    Monitor *m;
  public:
    Agent(int, Monitor *);
    void operator()(); 
};
//--------------------------------------------
Agent::Agent(int r, Monitor *mon) : runs(r), m(mon){}
//--------------------------------------------
void Agent::operator()()
{
   for(int i=0;i<runs; i++)
   {
      int ingreds = rand() % 3;
      m->newIngredient(ingreds);      
   }
  m->finishSim();      
}
//***************************************************
int main(int argc, char **argv)
{
   Monitor m;
   Smoker *s[3];
   unique_ptr<thread> tp[4];
   
   for(int i=0;i<3;i++)
   { 
     s[i] = new Smoker(i, &m);
     tp[i] = make_unique< thread >(ref(*s[i]));
   }
   Agent a(atoi(argv[1]), &m);
   tp[3] = make_unique< thread >(ref(a));
   
   for(int i=0;i<4;i++)
     tp[i]->join();
   
   return EXIT_SUCCESS;
}
