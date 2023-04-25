/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : October 2019
 License       : Released under the GNU GPL 3.0
 Description   : Class that solves the ABA problem
 To compile    : Cannot compile a template separately
 ============================================================================
 */

#include<memory>
#include<limits>
#include<atomic>
#include<functional>

#include<mutex>  // for dump_lists debugging utility function
#include<string>
//---------------------------------------
template < typename T > class queue_lock_free_unbound
{
private:

  template < typename W > struct Node
  {
    W data;
    std::atomic < Node<T> * >next;   
    std::atomic < Node<T> * >delNext;
  };

  std::atomic < Node<T> * >head;
  std::atomic < Node<T> * >tail;
  std::atomic < Node<T> * >freeHead;
  std::atomic < Node<T> * >freeTail;
  std::atomic < int >numItems = 0;
  std::atomic < int >numFreeNodes = 0;

  std::atomic < int >numContestedFreeNodes=0;
  std::atomic < int >numContestedItems=0;

  std::atomic < int >numPoppers=0;
  std::atomic < int >numPushers=0;

  Node<T> * getFree ();
  void makeFree (Node<T> *);
public:
  bool empty ();
  int size ();
  void enque (T &);
  T deque ();
  void dump_lists (std::string s);
  
  queue_lock_free_unbound ();
  ~queue_lock_free_unbound ();
  queue_lock_free_unbound (queue_lock_free_unbound &) = delete; // delete the default copy constructor and copy assignment
  queue_lock_free_unbound & operator= (const queue_lock_free_unbound &) = delete;
};

//---------------------------------------
template < typename T > queue_lock_free_unbound < T >::queue_lock_free_unbound ()
{
  Node<T> *tmp = new Node<T>(); // sentinel for data queue
  tmp->next = nullptr;  
  head = tail = tmp;
   
  tmp = new Node<T>(); // sentinel for free nodes queue
  tmp->next = nullptr;  
  freeHead = freeTail = tmp;
}

//---------------------------------------
template < typename T > queue_lock_free_unbound < T >::~queue_lock_free_unbound ()
{
  Node<T> *tmp = head.load();
  Node<T> *next;
  while(tmp != nullptr)
  {
     next = tmp->next;
     delete tmp;   
     tmp = next;
  }

  tmp = freeHead.load();    
  while(tmp != nullptr)
  {
     next = tmp->next;
     delete tmp;   
     tmp = next;
  }    
}

//---------------------------------------
template < typename T > queue_lock_free_unbound<T>::Node<T>*  queue_lock_free_unbound < T >::getFree ()
{
  Node<T>* newPtr;
  Node<T>* succPtr;
  numPushers++;
  while (true)
    {
      newPtr = freeHead.load()->delNext;
      if (numFreeNodes == 0 || newPtr == nullptr)
        {            
            if(numPoppers==0 && numContestedFreeNodes>0) // is there one we can retrieve from previous released nodes?
               {
                int tmp = numContestedFreeNodes;
                while(!numContestedFreeNodes.compare_exchange_weak(tmp, 0)); // make sure numContestedItems is modified atomically
                numFreeNodes+= tmp;
                continue;
               }
            else   
               {
                // return a newly allocated node
                numPushers--; 
                return new Node<T>();
               }
        }
      if (freeHead.load()->delNext == newPtr)
        {
          succPtr = newPtr->delNext;
          if (succPtr == nullptr)
            if (!freeTail.compare_exchange_strong (newPtr, freeHead))   // free list became empty
              continue;
          if (freeHead.load()->delNext.compare_exchange_strong (newPtr, succPtr))
            break;
        }

    }
  numPushers--;  
  numFreeNodes--;
  return newPtr;
}

//---------------------------------------
template < typename T > void queue_lock_free_unbound < T >::makeFree (Node<T>* currPtr)
{
  currPtr->delNext = nullptr;
  while (true)
    {
      Node<T>* tmpPtr = freeTail.load ();
      Node<T>* nextPtr = freeTail.load()->delNext;
      if (tmpPtr == freeTail.load ())
        {
          if (nextPtr == nullptr)
            {
              if (tmpPtr->delNext.compare_exchange_strong (nextPtr, currPtr))
                {
                  freeTail.compare_exchange_strong (tmpPtr, currPtr);   // if it fails next iteration will advance it
                  if(numPoppers==0)
                  {
                      int tmp = numContestedFreeNodes;
                      while(!numContestedFreeNodes.compare_exchange_weak(tmp, 0)); // make sure numContestedFreeNodes is modified atomically
                      tmp++; // for the newly freed node
                      numFreeNodes+= tmp;                      
                  }
                  else
                      numContestedFreeNodes++;
                  return;
                }
            }
          else
            freeTail.compare_exchange_strong (tmpPtr, nextPtr);
        }
    }
}

//---------------------------------------
template < typename T > bool queue_lock_free_unbound < T >::empty ()
{
  return numItems == 0;
}

//---------------------------------------
template < typename T > int queue_lock_free_unbound < T >::size ()
{
  return numItems;
}

//---------------------------------------
template < typename T > void queue_lock_free_unbound < T >::enque (T & i)
{
  Node<T>* newPtr;
  newPtr = getFree ();
  newPtr->data = i;
  newPtr->next = nullptr;

  while (true)
    {
      Node<T>* lastPtr = tail.load ();
      Node<T>* nextPtr = tail.load()->next;   // should be = nullptr
      if (lastPtr == tail.load ())
        {
          if (nextPtr == nullptr)
            {
              if (tail.load()->next.compare_exchange_strong (nextPtr, newPtr))
                {
                  tail.compare_exchange_strong (lastPtr, newPtr);
                  numContestedItems++; // for the newly enqueued node

                  if(numPushers==0)
                  {
                      int tmp = numContestedItems;
                      while(!numContestedItems.compare_exchange_weak(tmp, 0)); // make sure numContestedItems is modified atomically
                      numItems+= tmp;                      
                  }

                  return;
                }
            }
          else
            tail.compare_exchange_strong (lastPtr, nextPtr);
        }
    }
}

//---------------------------------------
template < typename T > T queue_lock_free_unbound < T >::deque ()
{
  Node<T>* poppedPtr;
  Node<T>* nextPtr;
  numPoppers++;
  while (true)
    {
      poppedPtr = head.load()->next;
      if (numItems == 0 || poppedPtr == nullptr)
        {
          numPoppers--;  
          std::this_thread::yield ();
          numPoppers++;
          continue;
        }
      if (head.load()->next == poppedPtr)
        {
          nextPtr = poppedPtr->next;
          if (nextPtr == nullptr)
            if (!tail.compare_exchange_strong (poppedPtr, head))        // tail should point to where head points to when queue becomes empty
              continue;
          if (head.load()->next.compare_exchange_strong (poppedPtr, nextPtr))
            break;
        }
    }
  T tmp = std::move (poppedPtr->data);
  numPoppers--;
  makeFree (poppedPtr);
  numItems--;
  return std::move (tmp);
}

//---------------------------------------
std::mutex l;
template < typename T > void queue_lock_free_unbound < T >::dump_lists (std::string s)
{
  l.lock ();
  std::cout << s << std::endl;
  std::cout << "Items: " << numItems << std::endl;
  std::cout << "ContestedItems: " << numContestedItems << std::endl;
  std::cout << "FreeItems: " << numFreeNodes << std::endl;
  std::cout << "ContestedFree: " << numContestedFreeNodes << std::endl;
  std::cout << "List ";
  Node<T> *tmp = head.load();
  while(tmp != nullptr)
  {
     std::cout << tmp->data << " ";
     tmp = tmp->next;
  }

  std::cout << "\nFree ";
  tmp = freeHead;    
  while(tmp != nullptr)
  {
     std::cout << tmp->data << " ";
     tmp = tmp->next;
  }

  l.unlock ();
}
