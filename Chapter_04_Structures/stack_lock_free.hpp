/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : October 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : 
 ============================================================================
 */

#include <atomic>
#include <istream>
#include <limits>
#include <memory>
#include <functional>

class empty_stack : std::exception
{
public:  
  virtual const char* what() const noexcept {return "Stack is empty";};
};

//---------------------------------------
template <typename T>
class stack_lock_free
{
private:
template <typename W>
    struct Node{
        W data;
        Node<W> * next;
    };
    
   std::atomic<Node<T> *> head;
    
public:
    T pop();
    void push(const T&);
    stack_lock_free(){ head.store(nullptr);}
    stack_lock_free(const stack_lock_free &) = delete;
    stack_lock_free & operator=(const stack_lock_free &) = delete;
};

//---------------------------------------
template<typename T>
void stack_lock_free<T>::push(const T&i)
{
    Node<T> *newNode = new Node<T>;
    newNode->data = i;
    newNode->next = head.load();
    while(! head.compare_exchange_weak( newNode->next, newNode) );    
}

//---------------------------------------
template<typename T>
T stack_lock_free<T>::pop()
{
    Node<T> *candidate = head.load();
    while(candidate != nullptr && !head.compare_exchange_weak(candidate, candidate->next));
    if(candidate == nullptr) throw empty_stack();
    T tmp = candidate->data;
    delete candidate;
    return std::move(tmp);
}

