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

#include<mutex>
#include<memory>
#include<limits>
#include<atomic>
#include<functional>

//---------------------------------------
template<typename T>
class list_fine{
private:
    
template<typename W>
    struct Node{
        W data;
        size_t key;
        std::unique_ptr<Node<W>> next;
        std::mutex lck;
    };

    std::unique_ptr<Node<T>> head;
    std::atomic<int> numItems=0;
public:
    bool empty();
    int size();
    bool insert(T &);
    bool erase(T &);
    void dump();
    
    list_fine();
    list_fine(list_fine &) = delete;   // delete the default copy constructor and copy assignment
    list_fine & operator=(const list_fine &) = delete;
};

//---------------------------------------
template<typename T> list_fine<T>::list_fine()
{
   std::unique_ptr<Node<T>> smallestSentinel = std::make_unique<Node<T>>();    
   smallestSentinel->key=0;
   std::unique_ptr<Node<T>> biggestSentinel = std::make_unique<Node<T>>();    
   biggestSentinel->key = std::numeric_limits<size_t>::max();
   smallestSentinel->next = std::move(biggestSentinel);
   head = std::move(smallestSentinel);
}

//---------------------------------------
template<typename T> bool list_fine<T>::empty()
{
    return numItems==0;
}

//---------------------------------------
template<typename T> int list_fine<T>::size()
{
    return numItems;
}

//---------------------------------------
template<typename T> bool list_fine<T>::insert(T & i)
{
    std::hash<T> h;   
    size_t ikey = h(i);
    if(ikey==0) ikey++;  // make sure there is no interference with the sentinels
    else if(ikey== std::numeric_limits<size_t>::max()) ikey--;
    
    Node<T> *curr, *prev;
    head->lck.lock();
    prev = head.get();
    
    prev->next->lck.lock();
    curr = head->next.get();
    while( curr->key < ikey || ( curr->key == ikey && curr->data < i))
    {
        prev->lck.unlock();
        prev = curr;        
        curr = curr->next.get();
        curr->lck.lock();
    }
    bool res=true;
    if(curr -> data != i) // item does not exist in the list
    {
      std::unique_ptr<Node<T>> newNode = std::make_unique<Node<T>>();
      newNode->key = ikey;
      newNode->data= i;    
      newNode->next = std::move(prev->next);
      prev->next = std::move(newNode);
      numItems++;        
    }
    else
        res=false;
   
    curr->lck.unlock(); 
    prev->lck.unlock(); // unlocked last
    return res;
}

//---------------------------------------
template<typename T> bool list_fine<T>::erase(T &i) 
{
    std::hash<T> h;   
    size_t ikey = h(i);
    if(ikey==0) ikey++;  // make sure there is no interference with the sentinels
    else if(ikey== std::numeric_limits<size_t>::max()) ikey--;
    
    Node<T> *curr, *prev;
    head->lck.lock();
    prev = head.get();
    
    prev->next->lck.lock();
    curr = head->next.get();
    while( curr->key < ikey || ( curr->key == ikey && curr->data < i))
    {
        prev->lck.unlock();
        prev = curr;        
        curr = curr->next.get();
        curr->lck.lock();
    }
    bool res=true;
    if(curr -> data == i  && curr->key != std::numeric_limits<size_t>::max()) 
    {
       prev->next = std::move( curr->next);
       numItems--;
    }
    else
       res=false;
   
    curr->lck.unlock(); 
    prev->lck.unlock(); // unlocked last
    return res;
}
//---------------------------------------
template<typename T>
void list_fine<T>::dump()
{
    size_t lastKey = std::numeric_limits<size_t>::max();
    Node<T> *curr = head->next.get();
    while(curr->key < lastKey)
    {
        std::cout << curr->data << " ";
        curr = curr->next.get();
    }
    std::cout << std::endl;
}

