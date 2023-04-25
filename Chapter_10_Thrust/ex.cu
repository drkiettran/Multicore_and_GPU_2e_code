#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

using namespace std;

void print(char c, thrust::host_vector<int> &v)
{
   cout << "Version " << c << " : ";    
   thrust::copy(v.begin(), v.end(), ostream_iterator<int>(cout, ", "));
   cout << endl;
}

int main ()
{
  thrust::host_vector < int >h_data (10, 0);        // all set to 0
  thrust::host_vector < int >h_add (10);
  thrust::sequence (h_add.begin (), h_add.end (), 10, 10);      // 10, 20, 30
  
  h_data.insert (h_data.begin (), h_add.begin (), h_add.end ());
  print('A', h_data);
  
  h_data.erase(h_data.begin()+12, h_data.end());
  print('B', h_data);
  
  thrust::fill(h_add.begin(), h_add.end(), 100);
  print('C', h_add);
  
  thrust::copy(h_data.begin(), h_data.begin()+4, h_add.begin());
  print('D', h_add);  
  return 0;
  
}
