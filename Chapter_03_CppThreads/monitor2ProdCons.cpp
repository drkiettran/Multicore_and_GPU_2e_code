/*
 ============================================================================
 Author        : G. Barlas
 Version       : 2.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake monitor2ProdCons.pro; make
 ============================================================================
 */
#include <thread>
#include <condition_variable>
#include <mutex>
#include <iostream>
#include <unistd.h>
#include <queue>
#include "semaphore.h"

using namespace std;

const int BUFFSIZE = 100;
//************************************************************

template<typename T>
class Monitor {
private:
    mutex l;
    condition_variable full, empty;
    queue<T *> emptySpotsQ;
    queue<T *> itemQ;
    T *buffer;
public:
    T* canPut();
    T* canGet();
    void donePutting(T *x);
    void doneGetting(T *x);
    Monitor(int n = BUFFSIZE);
    ~Monitor();
};
//-----------------------------------------

template<typename T>
Monitor<T>::Monitor(int n) {
    buffer = new T[n];
    for(int i=0;i<n;i++)
        emptySpotsQ.push(&buffer[i]);
}
//-----------------------------------------

template<typename T>
Monitor<T>::~Monitor() {
    delete []buffer;
}
//-----------------------------------------

template<typename T>
T* Monitor<T>::canPut() {
    unique_lock< mutex > ul(l);
    while (emptySpotsQ.size() == 0)
        full.wait(ul);
    T *aux = emptySpotsQ.front();
    emptySpotsQ.pop();
    return aux;
}
//-----------------------------------------

template<typename T>
T* Monitor<T>::canGet() {
    unique_lock< mutex > ul(l);
    while (itemQ.size() == 0)
        empty.wait(ul);
    T* temp = itemQ.front();
    itemQ.pop();
    return temp;
}
//-----------------------------------------

template<typename T>
void Monitor<T>::donePutting(T *x) {
    lock_guard< mutex > lg(l);
    itemQ.push(x);
    empty.notify_one();
}
//-----------------------------------------

template<typename T>
void Monitor<T>::doneGetting(T *x) {
    lock_guard< mutex > lg(l);
    emptySpotsQ.push(x);
    full.notify_one();
}
//************************************************************

template<typename T>
class Producer  {
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
        T* aux = mon->canPut();
        *aux = item;
        mon->donePutting(aux);
    }
}
//---------------------------------------  

template<typename T>
class Consumer  {
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
        T* aux = mon->canGet();
        T item = *aux; // take the item out
        mon->doneGetting(aux);
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
    cout << i;
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
