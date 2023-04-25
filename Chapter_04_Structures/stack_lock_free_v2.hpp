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
template < typename T > class stack_lock_free
{
private:
  template < typename W > struct Node
  {
    W data;
    Node < W > *next;
  };

  std::atomic < Node < T > *>head;
  std::atomic < Node < T > *>freeNodes;

//----------------------
  Node < T > *getFree ()
  {
    Node < T > *tmp = freeNodes;
    while (tmp != nullptr && !freeNodes.compare_exchange_weak (tmp, tmp->next));
    if (tmp == nullptr)
      tmp = new Node < T > ();
    return tmp;
  }
//-----------------------
  void releaseNode (Node < T > *n)
  {
    n->next = freeNodes;
    while (!freeNodes.compare_exchange_weak (n->next, n));
  }
//-----------------------
  
public:
  bool pop (T &);
  void push (const T &);
  stack_lock_free ()
  {
    head.store (nullptr);
    freeNodes.store (nullptr);
  }
  ~stack_lock_free ();
  stack_lock_free (const stack_lock_free &) = delete;
  stack_lock_free & operator= (const stack_lock_free &) = delete;
};

//---------------------------------------
template < typename T > stack_lock_free < T >::~stack_lock_free ()
{
  Node < T > *tmp, *aux;
  tmp = freeNodes;
  while (tmp != nullptr)
    {
      aux = tmp->next;
      delete tmp;
      tmp = aux;
    }
  tmp = head;
  while (tmp != nullptr)
    {
      aux = tmp->next;
      delete tmp;
      tmp = aux;
    }
}

//---------------------------------------
template < typename T > void stack_lock_free < T >::push (const T & i)
{
  Node < T > *newNode = getFree ();
  newNode->data = i;
  newNode->next = head.load ();
  while (!head.compare_exchange_weak (newNode->next, newNode));
}

//---------------------------------------
template < typename T > bool stack_lock_free < T >::pop (T &i)
{
  Node < T > *candidate = head.load ();
  while (candidate != nullptr && !head.compare_exchange_weak (candidate, candidate->next));
  if (candidate == nullptr)
    return false;
  i = std::move(candidate->data);
  releaseNode (candidate);
  return true;
}
