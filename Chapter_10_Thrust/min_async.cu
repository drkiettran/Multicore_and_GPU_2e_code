/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : August 2020
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc min_async.cu -o min_async
 ============================================================================
 */
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>

#include <stdlib.h>
#include <future>

using namespace std;
//***************************************************
struct functor
{
  __host__ __device__ float operator () (const float &x) const
  {
    return x * x;
  }
};

//***************************************************
template < typename Iterator, typename Functor, typename ResultPtr > __global__ void frontEnd (Iterator xs, Iterator ys, int N, Functor f, ResultPtr idx)
{
  thrust::transform (thrust::cuda::par, xs, xs + N, ys, f);

  *idx = thrust::min_element (thrust::cuda::par, ys, ys + N) - ys;
}

//***************************************************
int main (int argc, char **argv)
{
  float st, end;
  st = atof (argv[1]);
  end = atof (argv[2]);
  int dataPoints = atoi (argv[3]);
  float step = (end - st) / dataPoints;

  thrust::device_vector < float >d_x (dataPoints);
  thrust::device_vector < float >d_y (dataPoints);
  thrust::sequence (d_x.begin (), d_x.end (), st, step);

  // first way
  thrust::device_vector < int >d_res (1);
  frontEnd <<< 1, 1 >>> (d_x.begin (), d_y.begin (), dataPoints, functor (), d_res.data ());
  cudaDeviceSynchronize ();
  cout << "Function minimum over [" << st << "," << end << "] occurs at " << d_x[d_res.data ()[0]] << endl;

  // second way
  future < long >res = std::async (std::launch::async,[&](){
                                   functor f; thrust::transform (d_x.begin (), d_x.end (), d_y.begin (), f); return thrust::min_element (d_y.begin (), d_y.end ()) - d_y.begin ();}
  );
  int idx = res.get ();
  cout << "Function minimum over [" << st << "," << end << "] occurs at " << d_x[idx] << endl;
  return 0;
}
