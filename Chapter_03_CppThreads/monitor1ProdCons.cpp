/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake monitor1ProdCons.pro; make
 ============================================================================
 */
#include <thread>
#include <condition_variable>
#include "semaphore.h"
#include <mutex>
#include <iostream>
#include <unistd.h>

using namespace std;

const int BUFFSIZE = 100;
//************************************************************

template<typename T>
class Monitor {
private:
    mutex l;
    condition_variable full, empty;
    int in, out;
    int N;
    int count;
    T *buffer;
public:
    void put(T);
    T get();
    Monitor(int n = BUFFSIZE);
    ~Monitor();
};
//-----------------------------------------

template<typename T>
Monitor<T>::Monitor(int n) {
    buffer = new T[n];
    N = n;
    count = 0;
    in = out = 0;
}
//-----------------------------------------

template<typename T>
Monitor<T>::~Monitor() {
    delete []buffer;
}
//-----------------------------------------

template<typename T>
void Monitor<T>::put(T i) {
    unique_lock <mutex> ul(l);
    while (count == N)
        full.wait(ul);
    buffer[in] = i;
    in = (in + 1) % N;
    count++;
    empty.notify_one();
}
//-----------------------------------------

template<typename T>
T Monitor<T>::get() {
    unique_lock <mutex> ul(l);
    while (count == 0)
        empty.wait(ul);
    T temp = buffer[out];
    out = (out + 1) % N;
    count--;
    full.notify_one();
    return temp;
}
//************************************************************

template<typename T>
class Producer {
private:
    static semaphore numProducts;
    int ID;
    static Monitor<T> *mon;
public:
    static T(*produce)();
    static void initClass(int numP, Monitor<T> *m, T(*prod)());

    Producer<T>(int i) : ID(i) {}
    void operator()();
};
//---------------------------------------
template<typename T> semaphore Producer<T>::numProducts;
template<typename T> Monitor<T> * Producer<T>::mon;
template<> int (*Producer<int>::produce)() = NULL;
//---------------------------------------

template<typename T> void Producer<T>::initClass(int numP, Monitor<T> *m, T(*prod)()) {
    mon = m;
    numProducts.release(numP);
    produce = prod;
}
//---------------------------------------  

template<typename T>
void Producer<T>::operator()() {
    while (numProducts.try_acquire()) {
        T item = (*produce)();
        mon->put(item);
    }
}
//---------------------------------------  

template<typename T>
class Consumer {
private:
    int ID;
    static Monitor<T> *mon;
    static semaphore numProducts;
public:
    static void (*consume)(T i);
    static void initClass(int numP, Monitor<T> *m, void (*cons)(T));

    Consumer<T>(int i) : ID(i) {}
    void operator()();
};
//---------------------------------------

template<typename T> semaphore Consumer<T>::numProducts;
template<typename T> Monitor<T> *Consumer<T>::mon;
template<> void (*Consumer<int>::consume)(int) = NULL;

//---------------------------------------

template<typename T> void Consumer<T>::initClass(int numP, Monitor<T> *m, void (*cons)(T)) {
    numProducts.release(numP);
    mon = m;
    consume = cons;
}
//---------------------------------------

template<typename T> void Consumer<T>::operator()() {
    while (numProducts.try_acquire()) {
        T item = mon->get(); // take the item out
        (*consume)(item);
    }
}
//---------------------------------------

int produce() {
    // to be implemented
    return 1;
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
    Monitor<int> m;

    Producer<int>::initClass(numP, &m, &produce);
    Consumer<int>::initClass(numP, &m, &consume);

    shared_ptr< Producer<int> > p[N];
    shared_ptr< Consumer<int> > c[M];
    unique_ptr< thread > t[N+M];
    
    for (int i = 0; i < N; i++) {
        p[i] = make_shared< Producer<int> >(i);
        t[i] = make_unique< thread >(ref(*p[i]));      
    }
    for (int i = 0; i < M; i++) {
        c[i] = make_shared< Consumer<int> >(i);
        t[i+N] = make_unique< thread >(ref(*c[i]));      
    }

    for (int i = 0; i < N+M; i++)
        t[i]->join();

    return 0;
}
