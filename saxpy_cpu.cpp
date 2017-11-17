#include "saxpy.h"

int main(void)
{
   std::cout << "N: " << N << std::endl;
   float *x = new float[N], *y = new float[N];

   for (size_t i = 0; i < N; i++) {
      x[i] = XVAL;
      y[i] = YVAL;
   }

   saxpy_timer t;
   for (size_t i = 0; i < N; ++i) {
      y[i] += AVAL * x[i];
   }

   double elapsed = t.elapsed_msec();
   std::cout << "Elapsed: " << elapsed << " ms" << std::endl;
   saxpy_verify(y);

   delete[] x;
   delete[] y;
   return 0;
}
