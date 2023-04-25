/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : August 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake terminProdCons2.pro; make
 ============================================================================
 */
#include <stdlib.h>
#include <thread>
#include <mutex>
#include <iostream>
#include <unistd.h>
#include <memory>
#include <stdio.h>
#include "semaphore.h"

using namespace std;

const int BUFFSIZE = 100;

template<typename T>
class Producer {
private:
    int ID;
    static semaphore *slotsAvail;
    static semaphore *resAvail;
    static mutex l1;
    static T *buffer;
    static volatile bool *exitFlag;
    static int in;
public:
    static T(*produce)();
    static void initClass(semaphore *s, semaphore *a, T *b, T(*prod)(), bool *e);

    Producer<T>(int i) : ID(i) {
    }
    void operator()();
};
//---------------------------------------
template<typename T> semaphore * Producer<T>::slotsAvail;
template<typename T> semaphore * Producer<T>::resAvail;
template<typename T> mutex Producer<T>::l1;
template<typename T> T * Producer<T>::buffer;
template<typename T> volatile bool *Producer<T>::exitFlag;
template<typename T> int Producer<T>::in = 0;
template<> int (*Producer<int>::produce)() = NULL;
//---------------------------------------

template<typename T>
void Producer<T>::initClass(semaphore *s, semaphore *a, T *b, T(*prod)(), bool *e) {
    slotsAvail = s;
    resAvail = a;
    buffer = b;
    produce = prod;
    exitFlag = e;
}
//---------------------------------------

template<typename T>
void Producer<T>::operator()() {
    while (*exitFlag == false) {
        T item = (*produce)();
        slotsAvail->acquire(); // wait for an empty slot in the buffer

        if (*exitFlag) return; // stop immediately on termination    

        l1.lock();
        buffer[in] = item; // store the item
        in = (in + 1) % BUFFSIZE; // update the in index safely
        l1.unlock();
        resAvail->release(); // signal resource availability
    }
}
//---------------------------------------  

template<typename T>
class Consumer {
private:
    int ID;
    static semaphore *slotsAvail;
    static semaphore *resAvail;
    static mutex l2;
    static T *buffer;
    static int numConsumers, numProducers;
    static int out;
    static volatile bool *exitFlag;
public:
    static bool (*consume)(T i);
    static void initClass(semaphore *s, semaphore *a, T* b, bool (*cons)(T), int N, int M, bool *e);
    Consumer<T>(int i) : ID(i) {}
    void operator()();
};
//---------------------------------------

template<typename T> semaphore * Consumer<T>::slotsAvail;
template<typename T> semaphore * Consumer<T>::resAvail;
template<typename T> mutex Consumer<T>::l2;
template<typename T> volatile bool *Consumer<T>::exitFlag;
template<typename T> T * Consumer<T>::buffer;
template<typename T> int Consumer<T>::out = 0;
template<typename T> int Consumer<T>::numConsumers;
template<typename T> int Consumer<T>::numProducers;
template<> bool (*Consumer<int>::consume)(int) = NULL;

//---------------------------------------

template<typename T> 
void Consumer<T>::initClass(semaphore *s, semaphore *a, T* b, bool (*cons)(T), int N, int M, bool *e) {
    slotsAvail = s;
    resAvail = a;
    consume = cons;
    buffer = b;
    numProducers = N;
    numConsumers = M;
    exitFlag = e;
}
//---------------------------------------

template<typename T> void Consumer<T>::operator()() {
    while (*exitFlag == false) {
        resAvail->acquire(); // wait for an available item

        if (*exitFlag) return; // stop immediately on termination

        l2.lock();
        T item = buffer[out]; // take the item out
        out = (out + 1) % BUFFSIZE; // update the out index
        l2.unlock();
        slotsAvail->release(); // signal for a new empty slot 

        if ((*consume)(item)) break; // time to stop?
    }

    // only the thread initially detecting termination reaches here
    *exitFlag = true;
    resAvail->release(numConsumers - 1);
    slotsAvail->release(numProducers);
}
//---------------------------------------

int produce() {
    // to be implemented
    int aux = rand();
    return aux;
}
//---------------------------------------

bool consume(int i) {
    // to be implemented
    cout << "@"; // just to show something is happening
    if (i % 10 == 0) return true;
    else return false;
}
//---------------------------------------

int main(int argc, char *argv[]) {
    if (argc == 1) {
        cerr << "Usage " << argv[0] << " #producers #consumers\n";
        exit(1);
    }

    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int *buffer = new int[BUFFSIZE];
    semaphore avail, buffSlots(BUFFSIZE);
    bool exitFlag = false;

    Producer<int>::initClass(&buffSlots, &avail, buffer, &produce, &exitFlag);
    Consumer<int>::initClass(&buffSlots, &avail, buffer, &consume, N, M, &exitFlag);

    unique_ptr<thread > thr[M+N]; 
    shared_ptr<Producer<int>> p[N];
    shared_ptr<Consumer<int>> c[M];
    
    for (int i = 0; i < N; i++) {
         p[i] = make_shared<Producer<int>>(i);
         thr[i] = make_unique<thread>(ref(*p[i]));
       }
     for (int i = 0; i < M; i++) {
         c[i] = make_shared<Consumer<int>>(i);
         thr[i+N] = make_unique<thread>(ref(*c[i]));
     }

    for (int i = 0; i < N+M; i++)
        thr[i]->join();

    delete [] buffer;
    return 0;
}
