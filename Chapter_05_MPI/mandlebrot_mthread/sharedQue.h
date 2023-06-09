#ifndef SHAREDQUE_H
#define SHAREDQUE_H

#include <condition_variable>
#include <mutex>
#include <list>
#include <stdio.h>

using namespace std;

const int QUESIZE=16;
//------------------------------------------------------
template<typename T>
class QueueMonitor
{
private:
    list<T> freeList;
    list<T> queue;

    mutex l;
    condition_variable full, empty;

public:
    QueueMonitor(int, T prealloc);
    T reserve(); // reserve an item for management. reserve and enque are supposed to be used in tandem
    void enque(T);
    T deque();       // deque and release are supposed to be used in tandem
    void release(T); // returns an item to the free list
    int availItems();
};
//------------------------------------------------------
template<typename T> QueueMonitor<T>::QueueMonitor(int s, T prealloc)
{
    for(int i=0;i<s;i++)
        freeList.push_back(prealloc+i);
}
//------------------------------------------------------
// reserves an item so that its fields can be populated before it is enqueued
template<typename T> T QueueMonitor<T>::reserve()
{
    unique_lock< mutex > ul(l);
    while(freeList.empty())
        full.wait(ul);
    T tmp = freeList.front();
    freeList.pop_front();
    return tmp;
}
//------------------------------------------------------
// this is non-blocking because the item has been reserved before
template<typename T> void QueueMonitor<T>::enque(T item)
{
    lock_guard<mutex> lg(l);
    queue.push_back(item);
    empty.notify_one();
}
//------------------------------------------------------
template<typename T> T QueueMonitor<T>::deque()
{
    unique_lock< mutex > ul(l);  
    while(queue.empty())
        empty.wait(ul);
    T tmp = queue.front();
    queue.pop_front();
    return tmp;
}
//------------------------------------------------------
template<typename T> void QueueMonitor<T>::release(T item)
{
    lock_guard<mutex> lg(l);
    freeList.push_back(item);
    full.notify_one();
}
//------------------------------------------------------
template<typename T> int QueueMonitor<T>::availItems()
{
    lock_guard<mutex> lg(l);
    return queue.size();
}

#endif // SHAREDQUE_H
