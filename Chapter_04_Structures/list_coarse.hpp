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
#include<functional>

//---------------------------------------
template<typename T>
class list_coarse{
private:
    
template<typename W>
    struct Node{
        W data;
        size_t key;
        std::unique_ptr<Node<W>> next;
    };

    std::unique_ptr<Node<T>> head;
    std::mutex lck;
    int numItems=0;
public:
    bool empty();
    int size();
    bool insert(T &);
    bool erase(T &);
    void dump();
    
    list_coarse();
    list_coarse(list_coarse &) = delete;   // delete the default copy constructor and copy assignment
    list_coarse & operator=(const list_coarse &) = delete;
};

//---------------------------------------
template<typename T> list_coarse<T>::list_coarse()
{
   std::unique_ptr<Node<T>> smallestSentinel = std::make_unique<Node<T>>();    
   smallestSentinel->key=0;
   std::unique_ptr<Node<T>> biggestSentinel = std::make_unique<Node<T>>();    
   biggestSentinel->key = std::numeric_limits<size_t>::max();
   smallestSentinel->next = std::move(biggestSentinel);
   head = std::move(smallestSentinel);
}

//---------------------------------------
template<typename T> bool list_coarse<T>::empty()
{
    std::lock_guard<std::mutex> lg(lck);    
    return numItems==0;
}

//---------------------------------------
template<typename T> int list_coarse<T>::size()
{
    std::lock_guard<std::mutex> lg(lck);    
    return numItems;
}

//---------------------------------------
template<typename T> bool list_coarse<T>::insert(T & i)
{
    std::lock_guard<std::mutex> lg(lck);    
    std::hash<T> h;
    
    size_t ikey = h(i);
    if(ikey==0) ikey++;  // make sure there is no interference with the sentinels
    else if(ikey== std::numeric_limits<size_t>::max()) ikey--;
    
    Node<T> *curr, *prev;
    prev = head.get();
    curr = head->next.get();
    while( curr->key < ikey || ( curr->key == ikey && curr->data < i))
    {
        prev = curr;
        curr = curr->next.get();
    }
    if(curr -> data == i) return false;  // item already exists in the list
   
    std::unique_ptr<Node<T>> newNode = std::make_unique<Node<T>>();
    newNode->key = ikey;
    newNode->data= i;    
    newNode->next = std::move(prev->next);
    prev->next = std::move(newNode);
    numItems++;
    return true;
}

//---------------------------------------
template<typename T> bool list_coarse<T>::erase(T &i) 
{
    std::lock_guard<std::mutex> lg(lck);
    std::hash<T> h;
    
    size_t ikey = h(i);
    if(ikey==0) ikey++;  // make sure there is no interference with the sentinels
    else if(ikey== std::numeric_limits<size_t>::max()) ikey--;
    
    Node<T> *curr, *prev;
    prev = head.get();
    curr = head->next.get();
    while( curr->key < ikey || ( curr->key == ikey && curr->data < i))
    {
        prev = curr;
        curr = curr->next.get();
    }
    if(curr -> data != i  || curr->key == std::numeric_limits<size_t>::max()) return false;
   
    prev->next = std::move( curr->next);
    numItems--;
    return true;
}
//---------------------------------------
template<typename T>
void list_coarse<T>::dump()
{
    std::lock_guard<std::mutex> lg(lck);
    size_t lastKey = std::numeric_limits<size_t>::max();
    Node<T> *curr = head->next.get();
    while(curr->key < lastKey)
    {
        std::cout << curr->data << " ";
        curr = curr->next.get();
    }
    std::cout << std::endl;
}

