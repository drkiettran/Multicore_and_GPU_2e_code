/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : July 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP k-ary tree traversal
 To build use  : g++ -fopenmp k-ary2.cpp -o k-ary2
 ============================================================================
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <math.h>
#include <omp.h>

using namespace std;

//====================================
template < typename T > struct Node
{
  T data;
  int parent;
  vector < int >children;
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
// private:
  int N;
  int k=0; // for k-ary tree
  int maxDepth;
  void preOrder_aux (int, int );
  unique_ptr < Node < T >[] > nodes;
};

//------------------------------------
template < typename T > KTree < T >::KTree (int n)
{
  nodes = make_unique < Node < T >[] > (n);
  nodes[0].parent=-1;
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
  nodes[a].children.push_back (b);
  nodes[b].parent = a;
  if(k < nodes[a].children.size()) k = nodes[a].children.size(); // get the tree degree when building it
}

//------------------------------------
template < typename T > void KTree < T >::preOrder ()
{
  int P = omp_get_num_procs();
  maxDepth = log( (2*P +1) * (k -1) + 1)/log( k )  -1;

//   cout << maxDepth << endl;
#pragma omp parallel
  {
#pragma omp single
#pragma omp taskgroup
    preOrder_aux (0, 0);
  }
}

//------------------------------------
template < typename T > void KTree < T >::preOrder_aux (int n, int depth)
{
  cout << nodes[n].data << endl;
#pragma omp taskloop grainsize(1) if(depth <= maxDepth)
  for (int i = 0; i < nodes[n].children.size (); i++)
    preOrder_aux (nodes[n].children[i], depth+1);
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

  int count = 0;

#pragma omp parallel
  {
#pragma omp single
    {
#pragma omp taskgroup task_reduction(+:count)
#pragma omp taskloop in_reduction(+:count)
      for (int i = 0; i < tree->N; i++)
        count += pred (tree->nodes[i].data);
    }
  }

  cout << "Count " << count << endl;

  return 0;
}
