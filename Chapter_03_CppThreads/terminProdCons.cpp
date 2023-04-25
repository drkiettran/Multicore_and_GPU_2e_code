/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : August 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake terminProdCons.pro; make
 ============================================================================
 */
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
    static semaphore * slotsAvail;
    static semaphore * resAvail;
    static mutex l1;
    static semaphore * numProducts;
    static T* buffer;
    static int in;
public:
    static T(*produce)();
    static void initClass(int numP, semaphore *s, semaphore *a, T* b, T(*prod)());
    Producer<T>(int i) : ID(i) {};
    void operator()();
};
//---------------------------------------
template<typename T> semaphore * Producer<T>::numProducts;
template<typename T> semaphore * Producer<T>::slotsAvail;
template<typename T> semaphore * Producer<T>::resAvail;
template<typename T> mutex Producer<T>::l1;
template<typename T> T* Producer<T>::buffer;
template<typename T> int Producer<T>::in = 0;
template<> int (*Producer<int>::produce)() = NULL;
//---------------------------------------

template<typename T> void Producer<T>::initClass(int numP, semaphore *s, semaphore *a, T* b, T(*prod)()) {
    numProducts = new semaphore(numP);
    slotsAvail = s;
    resAvail = a;
    buffer = b;
    produce = prod;
}
//---------------------------------------  

template<typename T>
void Producer<T>::operator()() {
    while (numProducts->try_acquire()) {
        T item = (*produce)();
        slotsAvail->acquire(); // wait for an empty slot in the buffer
        l1.lock();
        buffer[in] = item; // store the item
        in = (in + 1) % BUFFSIZE; // update the in index safely
        l1.unlock();
        resAvail->release(); // signal resource availability
    }
}
//---------------------------------------  

template<typename T>
class Consumer  {
private:
    int ID;
    static semaphore * slotsAvail;
    static semaphore * resAvail;
    static mutex l2;
    static T* buffer;
    static int out;
    static semaphore *numProducts;
public:
    static void (*consume)(T i);
    static void initClass(int numP, semaphore *s, semaphore *a, T* b, void (*cons)(T));
    Consumer<T>(int i) : ID(i) {};
    void operator()();
};
//---------------------------------------

template<typename T> semaphore * Consumer<T>::numProducts;
template<typename T> semaphore * Consumer<T>::slotsAvail;
template<typename T> semaphore * Consumer<T>::resAvail;
template<typename T> mutex Consumer<T>::l2;
template<typename T> T* Consumer<T>::buffer;
template<typename T> int Consumer<T>::out = 0;
template<> void (*Consumer<int>::consume)(int) = NULL;

//---------------------------------------

template<typename T> void Consumer<T>::initClass(int numP, semaphore *s, semaphore *a, T* b, void (*cons)(T)) {
    numProducts = new semaphore(numP);
    slotsAvail = s;
    resAvail = a;
    buffer = b;
    consume = cons;
}
//---------------------------------------

template<typename T> void Consumer<T>::operator()() {
    while (numProducts->try_acquire()) {
        resAvail->acquire(); // wait for an available item
        l2.lock();
        T item = buffer[out];  // take the item out
        out = (out + 1) % BUFFSIZE; // update the out index
        l2.unlock();
        slotsAvail->release(); // signal for a new empty slot 
        (*consume)(item);
    }
}
//---------------------------------------

int produce() {
    // to be implemented
    static mutex l;
    static int i=0;
    l.lock();
    int tmp=i++;
    l.unlock();
    return tmp;
}
//---------------------------------------

void consume(int i) {
    // to be implemented
}
//---------------------------------------

int main(int argc, char *argv[]) {
    if (argc == 1) {
        cerr << "Usage " << argv[0] << " #producers #consumers #iterations\n";
        exit(1);
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int numP = atoi(argv[3]);
    int *buffer = new int[BUFFSIZE];
        
    unique_ptr<thread > thr[M+N]; 
    
    semaphore avail(0);
    semaphore buffSlots(BUFFSIZE);

    Producer<int>::initClass(numP, &buffSlots, &avail, buffer, &produce);
    Consumer<int>::initClass(numP, &buffSlots, &avail, buffer, &consume);

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

    delete []buffer;
    return 0;
}
