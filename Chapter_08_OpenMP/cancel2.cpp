/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : July 2020
 License       : Released under the GNU GPL 3.0
 Description   : OpenMP thread affinity display
 To build use  : g++ -fopenmp places.cpp -o places
 ============================================================================
 */
#include<iostream>
#include<omp.h>

using namespace std;

int count = 0;

int main ()
{
cout << "cancel-var : " << omp_get_thread_num() << endl;
bool found=false;
#pragma omp parallel firstprivate(found)
  {
#pragma omp for
   for (int i = 0; i < 100; i++)
      {
#pragma omp cancel for if(found)

#pragma omp cancellation point for
          
         found = (i==10); 
#pragma omp atomic
            count++;
     
      }
    }
cout << count << endl;
return 0;
}
