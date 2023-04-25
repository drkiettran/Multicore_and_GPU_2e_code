/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : July 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP k-ary tree traversal
 To build use  : g++ -fopenmp k-ary.cpp -o k-ary
 ============================================================================
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <omp.h>

using namespace std;
  int count;
//====================================
template < typename T > struct Node
{
  T data;
    Node < T > *parent;
    vector < Node < T > *>children;
};

//====================================
// Node 0 is supposed to be the root
template < typename T > class KTree
{
public:
  KTree (int n);
  void setData (int i, T d);
  void addEdge (int a, int b);
  void preOrder ();
  int filterCount (bool pred (T));
// private:
//   int count;
  int N;
  void preOrder_aux (Node < T > *);
  void filterCount_aux (Node < T > *, bool pred (T));
  unique_ptr < Node < T >[] > nodes;
};

//------------------------------------
template < typename T > KTree < T >::KTree (int n)
{
  nodes = make_unique < Node < T >[] > (n);
  N = n;
}

//------------------------------------
template < typename T > void KTree < T >::setData (int i, T d)
{
  nodes[i].data = d;
}

//------------------------------------
template < typename T > void KTree < T >::addEdge (int a, int b)
{
  nodes[a].children.push_back (&nodes[b]);
  nodes[b].parent = &nodes[a];
}

//------------------------------------
template < typename T > void KTree < T >::preOrder ()
{
#pragma omp parallel
  {
#pragma omp single
#pragma omp taskgroup
    preOrder_aux (&nodes[0]);
  }
}

//------------------------------------
template < typename T > void KTree < T >::preOrder_aux (Node < T > *n)
{
  if (n == nullptr)
    return;
  cout << n->data << endl;
#pragma omp taskloop grainsize(1)
  for (int i = 0; i < n->children.size (); i++)
    {
//          printf("Thread ID #%i : %i %i\n", omp_get_thread_num(), n->data, i);
      preOrder_aux (n->children[i]);
    }
}

//------------------------------------
template < typename T > int KTree < T >::filterCount (bool pred (T))
{
  count = 0;

#pragma omp parallel
  {
#pragma omp single
    {
#pragma omp taskgroup task_reduction(+:count)
        filterCount_aux (&nodes[0], pred);
    }
  }
  return count;
}

//------------------------------------
template < typename T > void KTree < T >::filterCount_aux (Node < T > *n, bool pred (T))
{
  if (n == nullptr)
    return;

  count += pred (n->data);

// #pragma omp taskloop grainsize(1) 
#pragma omp taskloop grainsize(1) in_reduction(+:count)
  for (int i = 0; i < n->children.size (); i++)
    {
//          printf("Thread ID #%i : %i %i\n", omp_get_thread_num(), n->data, i);
      filterCount_aux (n->children[i], pred);
    }
}

//====================================
// Expecting the first line to contain the number of nodes and the number of edges E
// E lines follow, each having the ID of the parent and the ID of the child. IDs start from 0
unique_ptr < KTree < int >>readTree (const char *fname)
{
  ifstream fin (fname);
  int N, E, a, b;
  fin >> N >> E;
  unique_ptr < KTree < int >>tree = make_unique < KTree < int > >(N);
  for (int i = 0; i < N; i++)
    tree->setData (i, i);
  while (E--)
    {
      fin >> a >> b;
      tree->addEdge (a, b);
    }
  fin.close ();
  return move (tree);
}

bool pred (int n)
{
  return n % 2 == 0;
}

//====================================
int main (int argc, char **argv)
{
  unique_ptr < KTree < int >>tree = readTree (argv[1]);
  tree->preOrder ();

  cout << "Count " << tree->filterCount (pred) << endl;


//   int count = 0;
// 
// #pragma omp parallel
//   {
// #pragma omp single
//     {
// #pragma omp taskgroup task_reduction(+:count)
// #pragma omp taskloop in_reduction(+:count)
//       for (int i = 0; i < tree->N; i++)
//         count += pred (tree->nodes[i].data);
//     }
//   }


  return 0;
}
