// Using NVidia Thrust API
// http://docs.nvidia.com/cuda/thrust/index.html
#include "saxpy.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include "cuda.h"
#include <string>

// Using functor idiom as recommended by http://docs.nvidia.com/cuda/thrust/index.html
struct saxpy_functor
{
   const float a_;
   saxpy_functor(float a) : a_(a) {}
   __host__  __device__ float operator()(const float& x, const float &y) const {
      return a_ * x + y;
   }
};

struct error_diff_functor
{
   const float answer_;
   error_diff_functor(float answer) : answer_(answer) {}
   __host__ __device__ float operator()(const float &y) {
      return fabs(y - answer_);
   }
};

template<class Iterator>
void dump_vector(const char *name, Iterator begin, Iterator end)
{
   std::cout << name << ": ";
   for (Iterator it = begin; it != end; ++it)
      std::cout << *it << " ";
   std::cout << " .." << std::endl;
}

template <class Vector, class SyncFunction>
int saxpy_main(Vector dummy, SyncFunction sync)
{
   const unsigned N = (1 << 26);
   const float XVAL = 3;
   const float YVAL = 4;
   const float AVAL = 1.5;

   std::cout << "N: " << N << std::endl;
   Vector x(N);
   Vector y(N);

   thrust::fill(x.begin(), x.end(), XVAL);
   thrust::fill(y.begin(), y.end(), YVAL);

   sync();
   saxpy_timer t;
   thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), saxpy_functor(AVAL));
   sync();

   double elapsed = t.elapsed_msec();

   std::cout << "Elapsed: " << elapsed << " ms" << std::endl;

   const auto answer = YVAL + AVAL * XVAL;
   auto err = thrust::transform_reduce(y.begin(), y.end(), error_diff_functor(answer), 0, thrust::plus<float>());
   std::cout << "Errors: " << err << std::endl;
   //dump_vector("y", y.end()-4, y.end());
   return 0;
}

int main(int argc, char *argv[])
{
   if (argc==2 && std::string("cpu") == argv[1]) {
      std::cout << "Using cpu" << std::endl;
      return saxpy_main(thrust::host_vector<float>(), [](){});
   } else {
      std::cout << "Using gpu" << std::endl;
      return saxpy_main(thrust::device_vector<float>(), [](){ cudaDeviceSynchronize(); });
   }
}

