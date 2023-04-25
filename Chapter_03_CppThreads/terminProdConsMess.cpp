/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : August 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake terminProdConsMess.pro; make
 ============================================================================
 */
#include <thread>
#include <mutex>
#include <iostream>
#include <unistd.h>
#include <memory>
#include <math.h>
#include "semaphore.h"

using namespace std;

const int BUFFSIZE = 10;
const double LOWERLIMIT = 0;
const double UPPERLIMIT = 10;
//--------------------------------------------
typedef struct Slice {
    double start;
    double end;
    int divisions;
} Slice;
//--------------------------------------------
double func(double x) {
    return fabs(sin(x));
}
//--------------------------------------------
// acts as a consumer
class IntegrCalc {
private:
    int ID;
    static semaphore *slotsAvail;
    static semaphore *resAvail;
    static mutex l2;
    static mutex resLock;
    static Slice *buffer;
    static int out;
    static double *result;
    static semaphore numProducts;
public:
    static void initClass(semaphore *s, semaphore *a, Slice *b, double *r);

    IntegrCalc(int i) : ID(i) {};
    void operator()();
};
//---------------------------------------

semaphore * IntegrCalc::slotsAvail;
semaphore * IntegrCalc::resAvail;
mutex IntegrCalc::l2;
mutex IntegrCalc::resLock;
Slice * IntegrCalc::buffer;
int IntegrCalc::out = 0;
double *IntegrCalc::result;

//---------------------------------------

void IntegrCalc::initClass(semaphore *s, semaphore *a, Slice *b, double *res) {
    slotsAvail = s;
    resAvail = a;
    buffer = b;
    result = res;
    *result = 0;
}
//---------------------------------------

void IntegrCalc::operator()() {
    while (1) {
        resAvail->acquire(); // wait for an available item
        l2.lock();
        int tmpOut = out;
        out = (out + 1) % BUFFSIZE; // update the out index
        l2.unlock();

        // take the item out
        double st = buffer[tmpOut].start;
        double en = buffer[tmpOut].end;
        int div = buffer[tmpOut].divisions;

        slotsAvail->release(); // signal for a new empty slot 

        if (div == 0) break; // exit

        //calculate area  
        double localRes = 0;
        double step = (en - st) / div;
        double x;
        x = st;
        localRes = func(st) + func(en);
        localRes /= 2;
        for(int i=1; i< div; i++)   {
            x += step;
            localRes += func(x);
          }
        localRes *= step;

        // add it to result
        resLock.lock();
        *result += localRes;
        resLock.unlock();
    }
}
//---------------------------------------

int main(int argc, char *argv[]) {
    if (argc == 1) {
        cerr << "Usage " << argv[0] << " #threads #jobs\n";
        exit(1);
    }
    int N = atoi(argv[1]);
    int J = atoi(argv[2]);
    Slice *buffer = new Slice[BUFFSIZE];
    semaphore avail, buffSlots(BUFFSIZE);
    int in = 0;
    double result;

    IntegrCalc::initClass(&buffSlots, &avail, buffer, &result);

    unique_ptr<thread > thr[N]; 
    
    shared_ptr<IntegrCalc> func[N];

    for (int i = 0; i < N; i++) {
         func[i] = make_shared<IntegrCalc>(i);
         thr[i] = make_unique<thread>(ref(*func[i]));
       }

       
    // main thread is responsible for handing out 'jobs'
    // It acts as the producer in this setup
    double divLen = (UPPERLIMIT - LOWERLIMIT) / J;
    double st, end = LOWERLIMIT;
    for (int i = 0; i < J; i++) {
        st = end;
        end += divLen;
        if (i == J - 1) end = UPPERLIMIT;

        buffSlots.acquire();
        buffer[in].start = st;
        buffer[in].end = end;
        buffer[in].divisions = 1000;
        in = (in + 1) % BUFFSIZE;
        avail.release();
    }

    // put termination sentinels in buffer
    for (int i = 0; i < N; i++) {
        buffSlots.acquire();
        buffer[in].divisions = 0;
        in = (in + 1) % BUFFSIZE;
        avail.release();
    }

    // wait for all threads to finish      
    for (int i = 0; i < N; i++)
        thr[i]->join();
    
    delete [] buffer;
    cout << "Result : " << result << endl;
    return 0;
}
