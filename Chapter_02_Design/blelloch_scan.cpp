/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : January 2022
 License       : Released under the GNU GPL 3.0
 Description   : Demonstration of Blelloch's scan algorithm. CLI parameter has to be a power of 2
 To compile    : g++ blelloch_scan.cpp -o blelloch_scan
 To run        : ./blelloch_scan 16  
 ============================================================================
 */
#include<iostream>
#include<cstdlib>

using namespace std;

int main (int argc, char **argv)
{
  int N = atoi (argv[1]);
  int d[N];
  for (int i = 0; i < N; i++)
    d[i] = i;


  // reduction phase
  int step = 2;
  int smallStep = 1;
  while (smallStep < N)
    {
      cout << "STEP " << step << " : ";
      for (int i = 0; i < N; i++)
        cout << d[i] << " ";
      cout << endl;

      for (int i = 0; i < N; i += step)
        {
          cout << i + smallStep - 1 << " " << i + step - 1 << endl;
          d[i + step - 1] += d[i + smallStep - 1];
        }

      smallStep = step;
      step *= 2;
    }
  cout << "AFTER " << step << " : ";
  for (int i = 0; i < N; i++)
    cout << d[i] << " ";
  cout << endl;

  d[N - 1] = 0;
  // down-sweep phase 
  step = smallStep;
  smallStep /= 2;
  while (smallStep > 0)
    {
      for (int i = 0; i < N; i += step)
        {
          int temp = d[i + smallStep - 1];
          d[i + smallStep - 1] = d[i + step - 1];
          d[i + step - 1] += temp;
        }

      step = smallStep;
      smallStep /= 2;
    }

  for (int i = 0; i < N; i++)
    cout << d[i] << " ";
  cout << endl;
  return 0;
}
