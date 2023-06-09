/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.1
 Last modified : September 2019
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake outliers.pro; make
 ============================================================================
 */
#include <QtConcurrentFilter>
#include <QList>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <functional>

using namespace std;

//******************************************
bool filterFunc (const int &x, const int &lower, const int &upper)
{
  if(x>upper)
    return false;
  if(x<lower)
    return false;
  return true;
}

//******************************************
int main (int argc, char *argv[])
{
  srand (time (0));
  int N = atoi (argv[1]);
  int l = atoi (argv[2]);
  int u = atoi (argv[3]);
  QList < int >data;
  for (int i = 0; i < N; i++)
    data.append ((rand () % 2000000) - 1000000);

  QList<int> out = QtConcurrent::blockingFiltered (data, bind(filterFunc, placeholders::_1, l, u));

  cout << "Filtered array is " << out.size() << " long\n";
  return 0;
}
