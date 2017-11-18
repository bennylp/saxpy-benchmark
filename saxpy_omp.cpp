#include "saxpy.h"
#include <omp.h>

int main() {
   float *x = new float[N], *y = new float[N];

   int g_num_threads = omp_get_num_threads();
#pragma omp parallel
   for (int i = 0; i < N; ++i) {
      int num_threads = omp_get_num_threads();
      g_num_threads = num_threads;
      x[i] = XVAL;
      y[i] = YVAL;
   }
   std::cout << "Number of threads: " << g_num_threads << std::endl;
   std::cout << "N: " << N << std::endl;

   saxpy_timer timer;
#pragma omp parallel
   {
      int num_threads = omp_get_num_threads();
      for (int i=omp_get_thread_num(); i<N; i+=num_threads)
         y[i] += AVAL * x[i];
   }

   auto elapsed = timer.elapsed_msec();
   std::cout << "Elapsed: " << elapsed << " ms\n";

   saxpy_verify(y);
   delete[] x;
   delete[] y;
   return 0;
}

