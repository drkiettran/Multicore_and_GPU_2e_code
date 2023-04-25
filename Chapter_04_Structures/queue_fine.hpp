/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.1
 Last modified : November 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : Cannot compile a template separately
 ============================================================================
 */

#include<mutex>
#include<exception>
#include<memory>
#include<atomic>

//---------------------------------------
template<typename T>
class queue{
private:
    
template<typename W>
    struct Node{
        W data;
        std::unique_ptr<Node<W>> next;
    };

    std::unique_ptr<Node<T>> head;
    Node<T> *tail;
    std::mutex hl, tl; // head-lock, tail-lock
    std::atomic<int> numItems=0;
public:
    bool empty();
    int size();
    void push_back(T &);
    T pop_front();
    queue();
    queue (queue &) = delete; // delete the default copy constructor and copy assignment
    queue & operator= (const queue &) = delete;
};

//---------------------------------------
template<typename T> queue<T>::queue()
{
   std::unique_ptr<Node<T>> sentinelNode = std::make_unique<Node<T>>();    
   head = std::move(sentinelNode);
   tail = head.get();
}

//---------------------------------------
template<typename T> bool queue<T>::empty()
{
    return numItems.load()==0;
}

//---------------------------------------
template<typename T> int queue<T>::size()
{
    return numItems.load();
}

//---------------------------------------
template<typename T> void queue<T>::push_back(T & i)
{
    std::unique_ptr<Node<T>> newNode = std::make_unique<Node<T>>();
    newNode->data = i;
    std::lock_guard<std::mutex> lg(tl);
    tail->next = std::move(newNode);
    tail = tail->next.get();
    numItems++;
}

//---------------------------------------
template<typename T> T queue<T>::pop_front() 
{
    std::lock_guard<std::mutex> lg(hl);

    int tmp=numItems;   
    while(tmp ==0  || !numItems.compare_exchange_strong(tmp, tmp-1));

    T temp = std::move(head->next->data);
    if(head->next->next==nullptr)
    {
        std::lock_guard<std::mutex> lg2(tl);
        if(numItems==0)
            tail = head.get();
    }
    head->next = std::move(head->next->next);                        
    return std::move(temp);
}
