/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : Cannot compile a template separately
 ============================================================================
 */
#include <mutex>
#include <atomic>
#include <memory>
#include <functional>
#include <limits>
#include <cstdlib>

template < class T > class list_lazy
{
private:
  template < class W > class Node
  {
  public:
    W data;
    size_t key;
    std::shared_ptr < Node < W >> next;
    volatile bool marked;
    std::mutex lck;
    Node ():marked {false}, next{nullptr}  {};
  };

  std::shared_ptr < Node < T >> head;
  std::atomic < int >numItems = 0;
  bool validate (std::shared_ptr < Node < T >> prev, std::shared_ptr < Node < T >> curr);

public:
  bool empty ();
  int size ();
  bool insert (T &);
  bool erase (T &);
  bool contains (T &);

  list_lazy ();
  list_lazy (list_lazy &) = delete;     // delete the default copy constructor and copy assignment
  list_lazy & operator= (const list_lazy &) = delete;
};

//---------------------------------------
template < typename T > bool list_lazy < T >::empty ()
{
  return numItems == 0;
}

//---------------------------------------
template < typename T > int list_lazy < T >::size ()
{
  return numItems;
}

//---------------------------------------
template < class T > list_lazy < T >::list_lazy ()
{
  std::shared_ptr < Node < T >> smallestSentinel = std::make_shared < Node < T >> ();
  smallestSentinel->key = 0;
  std::shared_ptr < Node < T >> biggestSentinel = std::make_shared < Node < T >> ();
  biggestSentinel->key = std::numeric_limits < size_t >::max ();
  smallestSentinel->next = biggestSentinel;
  head = smallestSentinel;
}

//---------------------------------------
template < class T > bool list_lazy < T >::insert (T & i)
{
  std::hash < T > h;
  size_t ikey = h (i);
  if (ikey == 0)
    ikey++;                     // make sure there is no interference with the sentinels
  else if (ikey == std::numeric_limits < size_t >::max ())
    ikey--;

  bool res;
  while (true)
    {
      std::shared_ptr < Node < T >> prev = atomic_load (&head);
      std::shared_ptr < Node < T >> curr = atomic_load (&(head->next));
      while (curr->key < ikey || (curr->key == ikey && curr->data < i))
        {
          prev = atomic_load (&curr);
          curr = atomic_load (&(curr->next));
        }
      std::lock_guard < std::mutex > lg1 (prev->lck);
      std::lock_guard < std::mutex > lg2 (curr->lck);
      if (validate (prev, curr))        // if false repeat from scratch
        {
          if (curr->key == ikey && curr->data == i)
            res = false;
          else
            {
              std::shared_ptr < Node < T >> newNode = std::make_shared < Node < T >> ();
              newNode->key = ikey;
              newNode->data = i;
              newNode->next = atomic_load (&(prev->next));
              atomic_store (&(prev->next), newNode);
              numItems++;

              res = true;
            }

          return res;
        }
    }
}


//---------------------------------------
template < class T > bool list_lazy < T >::erase (T & i)
{
  std::hash < T > h;
  size_t ikey = h (i);
  if (ikey == 0)
    ikey++;                     // make sure there is no interference with the sentinels
  else if (ikey == std::numeric_limits < size_t >::max ())
    ikey--;

  bool res;
  while (true)
    {
      std::shared_ptr < Node < T >> prev = atomic_load (&head);
      std::shared_ptr < Node < T >> curr = atomic_load (&(head->next));
      while (curr->key < ikey || (curr->key == ikey && curr->data < i))
        {
          prev = atomic_load (&curr);
          curr = atomic_load (&(curr->next));
        }
      std::lock_guard < std::mutex > lg1 (prev->lck);
      std::lock_guard < std::mutex > lg2 (curr->lck);

      if (validate (prev, curr))        // if false repeat from scratch
        {
          bool res = true;
          if (curr->data == i && curr->key != std::numeric_limits < size_t >::max ())
            {
              curr->marked = true;      // logical removal
              atomic_store (&(prev->next), curr->next); // physical removal
              numItems--;
            }
          else
            res = false;

          return res;
        }
    }
}

//---------------------------------------
template < class T > bool list_lazy < T >::contains (T & item)
{
  int key = int (item);
  std::shared_ptr < Node < T >> curr = atomic_load (&head);
  while (curr->key < key)
    curr = atomic_load (&(curr->next));
  return curr->key == key && !curr->marked;
}

//---------------------------------------
template < class T > bool list_lazy < T >::validate (std::shared_ptr < Node < T >> prev, std::shared_ptr < Node < T >> curr)
{
  return !prev->marked && !curr->marked && prev->next == curr;
}

