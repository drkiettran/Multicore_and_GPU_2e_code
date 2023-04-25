/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : October 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : Cannot compile a template separately
 ============================================================================
 */

#include <atomic>
#include <istream>
#include <memory>
#include <functional>

//---------------------------------------
template < typename T > class stack_lock_free_smart
{
private:
  template < typename W > struct Node
  {
    W data;
    std::shared_ptr<Node < W >> next;
  };

  std::shared_ptr<Node<T>> head;

  
public:
  bool pop (T &);
  void push (const T &);
  stack_lock_free_smart ()
  {
    head = nullptr;
  }
  stack_lock_free_smart (const stack_lock_free_smart &) = delete;
  stack_lock_free_smart & operator= (const stack_lock_free_smart &) = delete;
};

//---------------------------------------
template < typename T > void stack_lock_free_smart < T >::push (const T & i)
{
  std::shared_ptr<Node < T >> newNode = std::make_shared<Node<T>>();
  newNode->data = i;
  newNode->next = std::atomic_load(&head);
  while (!std::atomic_compare_exchange_weak (&head, &(newNode->next), newNode));
}

//---------------------------------------
template < typename T > bool stack_lock_free_smart < T >::pop (T &i)
{
  std::shared_ptr<Node < T >> candidate = std::atomic_load(&head);
  while (candidate != nullptr && !std::atomic_compare_exchange_weak (&head, &candidate, candidate->next));
  if (candidate == nullptr)
     return false;
  i = std::move(candidate->data);
  return true;
}
