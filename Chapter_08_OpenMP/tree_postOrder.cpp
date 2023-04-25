/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : August 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : 
 ============================================================================
 */

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>

using namespace std;

// template structure for a tree node
template < class T > struct Node
{
  T info;
  Node *left, *right;

    Node (int i, Node < T > *l, Node < T > *r):info (i), left (l), right (r){}
};

//---------------------------------------
// function stub for processing a node's data
template < class T > void process (T item)
{
#pragma omp critical
  cout << "Processing " << item << " by thread " << omp_get_thread_num () << endl;
}

//---------------------------------------
template < class T > void postOrder (Node < T > *n)
{
  if (n == NULL)
    return;

#pragma omp task
  postOrder (n->left);
#pragma omp task
  postOrder (n->right);
#pragma omp taskwait

  process (n->info);
}

//---------------------------------------
int main (int argc, char *argv[])
{
  // build a sample tree
  Node < int >*head = new Node < int >(1, NULL, NULL);
  head->left = new Node < int >(2, NULL, NULL);
  head->right = new Node < int >(3, NULL, NULL);
  head->right->right = new Node < int >(4, NULL, NULL);

#pragma omp parallel
  {
#pragma omp single
    {
      postOrder (head);
    }
  }

  return 0;
}
