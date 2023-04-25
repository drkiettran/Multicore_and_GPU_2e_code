/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.1
 Last modified : July 2020
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -std=c++17 -fopenmp linked_list.cpp -o linked_list
 ============================================================================
 */
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>
#include <memory>

using namespace std;

// template structure for a list's node
template <class T>
struct Node
{
  T info;
  shared_ptr<Node<T>> next;
};
//---------------------------------------
// Appends a value at the end of a list pointed by the head *h
template <class T>
void append (int v, shared_ptr<Node<T>> &h)
{
  shared_ptr<Node<T>> tmp = make_shared<Node<T>> ();
  tmp->info = v;
  tmp->next = nullptr;

  shared_ptr<Node<T>> aux = h;
  if (aux->next == nullptr)              // first node in list
    h->next = tmp;
  else
    {
      while (aux->next != nullptr)
        aux = aux->next;
      aux->next = tmp;
    }
}

//---------------------------------------
// function stub for processing a node's data
template <class T>
void process (shared_ptr<Node<T>> p)
{
#pragma omp critical
  cout << p->info << " by thread " << omp_get_thread_num () << endl;
}

//---------------------------------------
int main (int argc, char *argv[])
{
  // build a sample list
  shared_ptr<Node<int>> head = make_shared<Node<int>>();
  head->next=nullptr;
  append (1, head);
  append (2, head);
  append (3, head);
  append (4, head);
  append (5, head);

#pragma omp parallel
  {
#pragma omp single
    {
      shared_ptr<Node<int>> tmp = head->next;
      while (tmp != nullptr)
        {
#pragma omp task
          process (tmp);
          tmp = tmp->next;
        }

    }
  }

  return 0;
}
