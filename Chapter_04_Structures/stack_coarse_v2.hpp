/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : 
 ============================================================================
 */

#include<iostream>
#include<vector>
#include<condition_variable>
#include<mutex>
#include<exception>

template<typename T>
class stack{
private:
    std::vector<T> store;
    std::mutex l;
    std::condition_variable c;
public:
    bool empty();
    int size();
    T pop();
    void push(T &);
};

template<typename T> bool stack<T>::empty()
{
    std::lock_guard<std::mutex> lg(l);
    return store.empty();
}

template<typename T> int stack<T>::size()
{
    std::lock_guard<std::mutex> lg(l);
    return store.size();
}

template<typename T> void stack<T>::push(T & i)
{
    std::lock_guard<std::mutex> lg(l);
    store.push_back(i);
    c.notify_one();
}

template<typename T> T stack<T>::pop() 
{
    std::unique_lock<std::mutex> ul(l);;
    while(store.size()==0)
        c.wait(ul);
    T temp = std::move(store.at[store.size()-1]);
    store.pop_back();
    return std::move(temp);

}

