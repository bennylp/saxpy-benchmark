// Bulk: https://github.com/jaredhoberock/bulk
#include <iostream>
#include <cstdio>
#include <cassert>
#include <bulk/bulk.hpp>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include "saxpy.h"

struct error_diff_functor
{
   const float answer_;
   error_diff_functor(float answer) : answer_(answer) {}
   __host__ __device__ float operator()(const float &y) {
      return fabs(y - answer_);
   }
};

struct saxpy
{
  __host__ __device__
  void operator()(bulk::agent<> &self, float a, float *x, float *y)
  {
    int i = self.index();
    y[i] = a * x[i] + y[i];
  }
};

int main()
{
  std::cout << "N: " << N << std::endl;

  thrust::device_vector<float> x(N, XVAL);
  thrust::device_vector<float> y(N, YVAL);

  float a = AVAL;

  cudaDeviceSynchronize();
  saxpy_timer t;
  bulk::async(bulk::par(N), saxpy(), bulk::root.this_exec,
              a, thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(y.data()));
  cudaDeviceSynchronize();
  double elapsed = t.elapsed_msec();

  std::cout << "Elapsed: " << elapsed << " ms" << std::endl;

  const auto answer = YVAL + AVAL * XVAL;
  auto err = thrust::transform_reduce(y.begin(), y.end(), error_diff_functor(answer), 0, thrust::plus<float>());
  std::cout << "Errors: " << err << std::endl;
  return 0;
}
