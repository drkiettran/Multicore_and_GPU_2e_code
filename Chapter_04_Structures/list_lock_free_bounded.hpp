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

#include<memory>
#include<limits>
#include<atomic>
#include<functional>
#include<memory>

#include<mutex>
#include<assert.h>
std::mutex l;
void debugPrint (const char *s)
{
  l.lock ();
  std::cout << std::this_thread::get_id () << " " << s << std::endl;
  l.unlock ();
}

//---------------------------------------
template < typename T > class list_lock_free_bounded
{
private:

  const uint64_t NORMAL = 0;
  const uint64_t DELETED = 1;
  const uint64_t FREE = 2;

  template < typename W > struct Node
  {
    W data;
    size_t key;
    std::atomic < uint64_t > next;
  };

  std::atomic < uint64_t > head;
  std::atomic < uint64_t > numItems = 0;

  // queue for managing free nodes
  std::atomic < uint64_t > freePushers = 0;
  std::atomic < uint64_t > freePoppers = 0;
  std::atomic < uint64_t > numFreeNodes;        // for managing the free nodes list
  std::atomic < uint64_t > freeIn, freeOut;     // for managing the free nodes list

  //------------------------------------------
  // Pointer/index components:
  // MCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCIIIIIIIIIIIIIIII
  // M : marked
  // C : counter
  // I : index
  uint64_t idxMask;
  uint64_t cntMask;
  int idx_bits;
  int counter_bits;

  //------------------------------------------
  // Pointer/index management methods
  uint64_t getIdx (uint64_t i)
  {
    return i & idxMask;
  }
  uint64_t getCounter (uint64_t i)
  {
    return (i >> idx_bits) & cntMask;
  }
  uint64_t joinIdxCounter (uint64_t idx, uint64_t cnt)
  {
    return (cnt << idx_bits) | idx;
  }
  int getStatus (uint64_t i)
  {
    return (i >> 62) & 3;
  }
  uint64_t resetStatus (uint64_t i, uint64_t status)    // clears previous status
  {
    return (i & (~(uint64_t) 0) >> 2) | (status << 62);
  }
  //------------------------------------------

  void findLoc (uint64_t * prev, uint64_t * curr, size_t k, T & i);
  uint64_t getFree ();
  bool makeFree (uint64_t, char);
  Node < T > *store;
  std::unique_ptr < std::atomic<int>[] > freeNodes;

  int storeSize;                // size of array. Also used to indicate end of a list

public:

  bool empty ();
  uint64_t size ();
  bool insert (T &);
  bool erase (T &);
  void dump_lists (std::string, bool e = false);

  list_lock_free_bounded (uint64_t ss = 100);
  ~list_lock_free_bounded ();
  list_lock_free_bounded (list_lock_free_bounded &) = delete;   // delete the default copy constructor and copy assignment
  list_lock_free_bounded & operator= (const list_lock_free_bounded &) = delete;
};

//---------------------------------------
template < typename T > list_lock_free_bounded < T >::list_lock_free_bounded (uint64_t ss)
{
  uint64_t updatedData;
  ss += 2;                      // add 2 sentinel nodes 
  store = new Node < T >[ss];
  storeSize = ss;

  // list starts with two sentinel nodes
  Node < T > *smallestSentinel = store;
  smallestSentinel->key = 0;
  Node < T > *biggestSentinel = store + 1;
  biggestSentinel->key = std::numeric_limits < size_t >::max ();
  biggestSentinel->next = storeSize;
  smallestSentinel->next = 1;
  head = 0;

  // free queue init.
  freeNodes = std::make_unique < std::atomic<int>[] > (ss - 2);      // unused nodes
  freeIn = 0;
  freeOut = 0;
  numFreeNodes = ss - 2;
  for (int i = 0; i < ss - 2; i++)
    {
      freeNodes[i] = i + 2;
      updatedData = joinIdxCounter (storeSize, 0);
      updatedData = resetStatus (updatedData, FREE);
      store[i + 2].next = updatedData;
    }


  // calculate the array index partitioning scheme
  uint64_t i = ss;
  idx_bits = 0;
  while (i > 0)
    {
      idx_bits++;
      i >>= 1;
    }
  idxMask = ~(uint64_t) 0;
  idxMask >>= (64 - idx_bits);
  counter_bits = 62 - idx_bits;
  cntMask = (~(uint64_t) 0) >> (idx_bits + 2);
}

//---------------------------------------
template < typename T > list_lock_free_bounded < T >::~list_lock_free_bounded ()
{
  delete[]store;
}

//---------------------------------------
template < typename T > uint64_t list_lock_free_bounded < T >::getFree ()
{
  uint64_t relIdx, relData, relCounter;
  uint64_t succIdx, succData, succCounter;
  uint64_t sentinelData, sentinelIdx, sentinelCounter, updatedData;
  uint64_t tailData, tailIdx, tailCounter;

  int iter = 0;

  // make sure no pushers are running    
  while (true)
    {
      debugPrint ("c");

      while (freePushers > 0)
        {
          debugPrint ("f");

          std::this_thread::yield ();
        }
      freePoppers++;
      if (freePushers == 0)
      {
          uint64_t tmp = numFreeNodes;
          if (tmp != 0 && numFreeNodes.compare_exchange_strong (tmp, tmp-1))
              break;
          else    
           {
             freePoppers--;  
             std::this_thread::yield ();
           }
        }
      else
        freePoppers--;
    }


  uint64_t tmpIdx = freeOut.fetch_add(1);
  int nodeIdx = freeNodes[tmpIdx % (storeSize - 2)]; 
  freePoppers--;
  return nodeIdx;

}

//---------------------------------------
template < typename T > bool list_lock_free_bounded < T >::makeFree (uint64_t currIdx, char from)
{
  uint64_t currData, currCounter, updatedData, currStatus;

  assert (currIdx >= 2);
  int iter = 0;
char b[100];

  // make sure no poppers are running    
  while (true)
    {
      debugPrint ("d");

      while (freePoppers > 0)
        {
          debugPrint ("g");

          std::this_thread::yield ();
        }
      freePushers++;
      if (freePoppers == 0)
        break;
      else
        freePushers--;
    }

  currData = store[currIdx].next;
  currCounter = getCounter (currData);
  currStatus = getStatus (currData);

  iter = 0;
  updatedData = joinIdxCounter (storeSize, currCounter + 1);
  updatedData = resetStatus (updatedData, FREE);
  if(currStatus == DELETED)
  {
    if(store[currIdx].next.compare_exchange_strong (currData, updatedData))
    {
     uint64_t tmpIdx = freeIn.fetch_add(1);
     freeNodes[tmpIdx % (storeSize - 2)] = currIdx;
     numFreeNodes++;
     freePushers--;
     return true;
    }
    sprintf(b,"makefree failed to update next for %li",currIdx);
    debugPrint(b);
     freePushers--;
     return false;        
    
  }
    sprintf(b,"makefree failed because status is not correct for %li",currIdx);
    debugPrint(b);
     freePushers--;
     return false;        
}

//---------------------------------------
// Input : item to search for (i) and corresponding hash value (k)
// Returns : index of preceeding node (*pr) and index of node containing item i or an item bigger than i (*cu)
template < typename T > void list_lock_free_bounded < T >::findLoc (uint64_t * pr, uint64_t * cu, size_t k, T & i)
{
  bool isMarked;
  bool removeSuccess;
  uint64_t currData, currCounter, currIdx, updatedData;
  uint64_t succData, succCounter, succIdx;
  uint64_t prevData, prevCounter, prevIdx;
  int iter = 0;
  while (true)
    {
      debugPrint ("a");
      iter++;
      if (iter > numItems + numFreeNodes)
        {
          char buff[100];
          sprintf (buff, " infindloc for %i iter", iter);
          debugPrint (buff);
          this->dump_lists ("AMAN");
        }
      prevIdx = head;           // points to first sentinel
      prevData = store[prevIdx].next;
      currIdx = getIdx (prevData);      // first actual data node
      prevCounter = getCounter (prevData);
      removeSuccess = true;
      while (removeSuccess)
        {
          iter++;
          if (iter > numItems + numFreeNodes)
            {
              char buff[100];
              sprintf (buff, "\t in removeHelper for iter %i processing %li", iter, currIdx);
              debugPrint (buff);
              dump_lists ("REMOVEHELPER");
            }
          currData = store[currIdx].next;
          succIdx = getIdx (currData);
          currCounter = getCounter (currData);
          isMarked = (getStatus (currData) == DELETED);
          while (isMarked)
            {
              iter++;
              if (iter > numItems + numFreeNodes)
                {
                  char buff[100];
                  sprintf (buff, "\t in ismarked for iter %i", iter);
                  debugPrint (buff);
                }

              if (getStatus (store[prevIdx].next) != NORMAL)    // previous node is marked?
                {
                  removeSuccess = false;
                  char buff[100];
                  sprintf (buff, "\t in removeHelper FAIL 1");
                  debugPrint (buff);

                  break;
                }
              if (getIdx (store[currIdx].next) != succIdx)      // any change between current and next one
                {
                  removeSuccess = false;
                  char buff[100];
                  sprintf (buff, "\t in removeHelper FAIL 2");
                  debugPrint (buff);
                  break;
                }
              if (getStatus(store[succIdx].next) == FREE)      // is the success node marked as FREE?
                {
                  removeSuccess = false;
                  char buff[100];
                  sprintf (buff, "\t in removeHelper FAIL 3");
                  debugPrint (buff);
                  break;
                }

              updatedData = joinIdxCounter (succIdx, prevCounter + 1);
              removeSuccess = store[prevIdx].next.compare_exchange_strong (prevData, updatedData);      // bypass current node
              if (!removeSuccess)
                {
                  char buff[100];
                  sprintf (buff, "\t in removeHelper FAIL 4");
                  debugPrint (buff);

                  break;
                }
              prevData = updatedData;   // change was successful

              numItems--;
              assert (currIdx != getIdx (store[currIdx].next));
              if (!makeFree (currIdx, 'C'))
                {
                  removeSuccess = false;
                  break;
                }

              currIdx = succIdx;        // update data to continue traversal/removal if needed
              currData = store[currIdx].next;
              succIdx = getIdx (currData);
              currCounter = getCounter (currData);
              isMarked = getStatus (currData) == DELETED;
            }

          if (!removeSuccess)   // if something went wrong in the while loop, start over
            break;

          if (store[currIdx].key > k || (store[currIdx].key == k && store[currIdx].data >= i))  // time to stop?
            {
              if (store[prevIdx].next != prevData || store[currIdx].next != currData)   // something changed in list?
                {
                  removeSuccess = false;
                  break;        // sanity check                    
                }
              *pr = prevIdx;    // return indices
              *cu = currIdx;
              return;
            }
          prevIdx = currIdx;    // continue traversal
          prevData = currData;
          prevCounter = currCounter;
          currIdx = getIdx (store[currIdx].next);       // get "fresh" data instead of using succIdx
        }
    }
}


//---------------------------------------
template < typename T > bool list_lock_free_bounded < T >::empty ()
{
  return numItems == 0;
}

//---------------------------------------
template < typename T > uint64_t list_lock_free_bounded < T >::size ()
{
  return numItems;
}

//---------------------------------------
template < typename T > bool list_lock_free_bounded < T >::insert (T & i)
{
  std::hash < T > h;
  uint64_t newIdx, newCounter, newData;
  uint64_t prevIdx, prevCounter, prevData;
  uint64_t updatedData;
  uint64_t currIdx, currData, currCounter, currPtr;
  bool nodeCreated = false;
  size_t ikey = h (i);

  if (ikey == 0)
    ikey++;                     // make sure there is no interference with the sentinels
  else if (ikey == std::numeric_limits < size_t >::max ())
    ikey--;


  int iter = 0;
  while (true)
    {
      debugPrint ("b");

      findLoc (&prevIdx, &currIdx, ikey, i);
      prevData = store[prevIdx].next;
      currData = store[currIdx].next;
      iter++;
      if (iter > numItems + numFreeNodes)
        {
          char buff[100];
          sprintf (buff, "\t in insert for iter %i %li %li ", iter, prevIdx, currIdx);
          debugPrint (buff);
          std::this_thread::yield ();
        }

      if (store[currIdx].key == ikey && store[currIdx].data == i && getStatus(store[currIdx].next)==NORMAL)       // item exists
        {
          if (nodeCreated)      // if a node was reserved in a previous iteration, it must be released
            {
              updatedData = joinIdxCounter (storeSize, newCounter + 1);
              updatedData = resetStatus (updatedData, DELETED);

              assert (newIdx != getIdx (updatedData));
              char b[100];
              sprintf(b, "Before returning unused node %li while inserting %i (counter %li %li)", newIdx, i, getCounter(newData), getIdx(newData));
              dump_lists (b);
              assert(store[newIdx].next.compare_exchange_strong (newData, updatedData));
              // return to free
              assert(makeFree (newIdx, 'A'));
            }
          return false;
        }
      else
        {
          iter++;
          if(!nodeCreated)     // get a node and initialize it
            {
              nodeCreated = true;
              char b[100];
              sprintf(b, "Before getting a free node for inserting %i", i);
              dump_lists(b);
              newIdx = getFree ();
//                dump_lists("AFTER");

              newData = store[newIdx].next;
              store[newIdx].key = ikey;
              store[newIdx].data = i;
              updatedData = joinIdxCounter (storeSize, getCounter (newData) + 1);
              updatedData = resetStatus (updatedData, NORMAL);
              assert(store[newIdx].next.compare_exchange_strong (newData, updatedData));
              newData = updatedData;
          newCounter = getCounter (newData);
              sprintf(b, "Just got free node %li for inserting %i (counter %li %li)", newIdx, i, getCounter(newData), getIdx(newData));
              dump_lists (b);
              assert(newData == store[newIdx].next);
//               if(newData != store[newIdx].next)
//                  nodeCreated=false;
            }


          // update new node
          if (newIdx == currIdx)        // this can happen if between traversal and getting a free node, currIdx is removed from the list
            {
              dump_lists ("SKA");
              char bu[100];
              sprintf (bu, "CURR %li  NEW %li   V %i", currIdx, newIdx, i);
              debugPrint (bu);

              continue;
            }

          // check before applying change
          if (getStatus (store[currIdx].next) != NORMAL || getStatus (store[prevIdx].next) != NORMAL)   // if either nodes have changed
            {
              if (iter > 10)
                dump_lists ("F2");
              continue;
            }

 
          if (store[prevIdx].next != prevData)  // list has changed
            {
              if (iter > 10)
                {
                  char buff[100];
                  sprintf (buff, " %li %li %li\n", store[prevIdx].next.load (), prevData, newIdx);
                  debugPrint (buff);
                  dump_lists ("F3");
                }
              continue;
            }
 
          newCounter++;
          updatedData = joinIdxCounter (currIdx, newCounter);
          updatedData = resetStatus(updatedData, NORMAL);

         if(!store[newIdx].next.compare_exchange_strong (newData, updatedData))
         {
              if(store[newIdx].next == newData)
              {
                  store[newIdx].next=updatedData;
              }
              else
              {
              char b[100];
              sprintf(b, "Newly acquired node %li for inserting %i (counter %li %li) was modified", newIdx, i, getCounter(newData), getIdx(newData));
              dump_lists (b);
              assert(0); // this should never trigger
              }
         }
         newData = updatedData;
 
         // update previous node
          prevCounter = getCounter (prevData);
          updatedData = joinIdxCounter (newIdx, prevCounter + 1);
          assert (prevIdx != getIdx (updatedData));

          if (store[prevIdx].next.compare_exchange_strong (prevData, updatedData))
            {
              numItems++;
              return true;
            }
          else
          {
              char b[100];
              sprintf(b, "Failed to update previous for  unused node %li while inserting %i (counter %li %li)", newIdx, i, getCounter(newData), getIdx(newData));
              dump_lists (b);
          }
        }
    }
}

//---------------------------------------
template < typename T > bool list_lock_free_bounded < T >::erase (T & i)
{
  uint64_t currData, currCounter, currIdx, updatedData;
  uint64_t prevIdx, prevCounter, prevData;
  uint64_t succIdx, succData, succPtr, succCounter;

  std::hash < T > h;
  size_t ikey = h (i);
  if (ikey == 0)
    ikey++;                     // make sure there is no interference with the sentinels
  else if (ikey == std::numeric_limits < size_t >::max ())
    ikey--;

  int iter = 0;
  while (true)
    {
      debugPrint ("a");

      iter++;
      if (iter > numItems + numFreeNodes)
        {
          char buff[100];
          sprintf (buff, "\t in erase for iter %i", iter);
          debugPrint (buff);
        }

      findLoc (&prevIdx, &currIdx, ikey, i);
      currData = store[currIdx].next;
      prevData = store[prevIdx].next;

      if (currIdx != storeSize && store[currIdx].key == ikey && store[currIdx].data == i)       // matching node found?
        {
          succIdx = getIdx (currData);
          succData = store[succIdx].next;
          updatedData = resetStatus (currData, DELETED);
          assert (getStatus (updatedData) == DELETED);

          if (store[currIdx].next.compare_exchange_strong (currData, updatedData))      // mark for removal
            {
              assert (getStatus (updatedData) == DELETED);

              // Logical removal
              if (getStatus (prevData) != NORMAL)       // cannot use prevIdx if it is marked
                continue;

              prevCounter = getCounter (prevData);
              updatedData = joinIdxCounter (succIdx, prevCounter + 1);

              if (store[prevIdx].next.compare_exchange_strong (prevData, updatedData))  // if this succeeds, the node is removed
                {
                  // physical removal
                  assert (currIdx != storeSize);
                  if (!makeFree (currIdx, 'B'))
                    {
                      char buff[100];
                      sprintf (buff, "Trying to erase %li", currIdx);
                      dump_lists ("SKA2");
                      debugPrint (buff);
                      continue;
                    }
                  numItems--;
                }
              return true;
            }
          else
            continue;
        }
      else
        return false;           // no match
    }
}

//---------------------------------------
template < typename T > void list_lock_free_bounded < T >::dump_lists (std::string s, bool toerr)
{
  l.lock ();
  if (toerr)
    {

      int iter = 0;
      std::cerr << std::this_thread::get_id () << " " << s << std::endl;
      std::cerr << "List (" << numItems << ") ";
      uint64_t x = head;
      while (x < storeSize && iter < 15)
        {
          iter++;
          std::cerr << x << " ";
          x = getIdx (store[x].next);
        }
      if (iter == 15)
        std::cerr << "Error\n";
      std::cerr << "\nFree (=" << numFreeNodes << " Push:" << freePushers << " Pop:" << freePoppers << " In:" << freeIn % (storeSize - 2) << " Out:" << freeOut % (storeSize - 2) << " ) ";

      iter = 0;
      uint64_t tmp = freeOut % (storeSize - 2);
      while ((iter < numFreeNodes || (tmp!= freeIn % (storeSize - 2))) && iter < 15)
        {
          iter++;
          std::cerr << freeNodes[tmp] << " ";
          tmp = (tmp + 1) % (storeSize - 2);
        }
 
     std::cerr << "\nFree list raw: ";
     iter=0;
     while (iter < storeSize-2 && iter < 15)
        {
          if(iter== (freeOut % (storeSize-2))) std::cerr << ">";  
          std::cerr << iter << ":" << freeNodes[iter] << " ";
          iter++;
        }
 
      std::cerr << "\nData ";
      x = head;
      int count = 0;
      iter = 0;
      while (x < storeSize && iter < 15)
        {
          iter++;
          std::cerr << store[x].data << " ";
          x = getIdx (store[x].next);
          count++;
        }
      std::cerr << "\n";
      if (iter == 15)
        std::cerr << "Error\n";
      std::cerr << "(" << count - 2 << ")\n";
      x = 0;
      while (x < storeSize && x < 15)
        {
          char buff[100];
          sprintf (buff, "#%li V:%i N:%li C:%li M:%i\n", x, store[x].data, getIdx (store[x].next), getCounter (store[x].next), getStatus (store[x].next));
          std::cerr << buff;
          x++;
        }

    }
  else
    {

      int iter = 0;
      std::cout << std::this_thread::get_id () << " " << s << std::endl;
      std::cout << "List (" << numItems << ") ";
      uint64_t x = head;
      while (x < storeSize && iter < 15)
        {
          iter++;
          std::cout << x << " ";
          x = getIdx (store[x].next);
        }
      if (iter == 15)
        std::cout << "Error\n";
      std::cout << "\nFree (=" << numFreeNodes << " Push:" << freePushers << " Pop:" << freePoppers << " In:" << freeIn % (storeSize - 2) << " Out:" << freeOut % (storeSize - 2) << " ) ";
      iter = 0;
      uint64_t tmp = freeOut % (storeSize - 2);
      while ((iter < numFreeNodes || (tmp!= freeIn % (storeSize - 2))) && iter < 15)
        {
          iter++;
          std::cout << freeNodes[tmp] << " ";
          tmp = (tmp + 1) % (storeSize - 2);
        }
     std::cout << "\nFree list raw: ";
     iter=0;
     while (iter < storeSize-2 && iter < 15)
        {
          if(iter==(freeOut % (storeSize-2))) std::cout << ">";  
          std::cout << iter << ":" << freeNodes[iter] << " ";
          iter++;
        }

      std::cout << "\nData ";
      x = head;
      int count = 0;
      iter = 0;
      while (x < storeSize && iter < 15)
        {
          iter++;
          std::cout << store[x].data << " ";
          x = getIdx (store[x].next);
          count++;
        }
      std::cout << "\n";

      if (iter == 15)
        std::cout << "Error\n";
      std::cout << "(" << count - 2 << ")\n";
      x = 0;
      while (x < storeSize && x < 15)
        {
          printf ("#%li V:%i N:%li C:%li M:%i\n", x, store[x].data, getIdx (store[x].next), getCounter (store[x].next), getStatus (store[x].next));
          x++;
        }
    }
  l.unlock ();
}
