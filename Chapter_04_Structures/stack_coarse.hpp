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
#include<mutex>
#include<exception>

class empty_stack : std::exception
{
public:  
  virtual const char* what() const noexcept {return "Stack is empty";};
};

template<typename T>
class stack{
private:
    std::vector<T> store;
    std::mutex l;
public:
    bool empty();
    int size();
    void push(T &);
    T pop();
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
}

template<typename T> T stack<T>::pop() 
{
    std::lock_guard<std::mutex> lg(l);
    if(store.size()==0)
        throw empty_stack();
    T temp = std::move(store.at[store.size()-1]);
    store.pop_back();
    return std::move(temp);

}

