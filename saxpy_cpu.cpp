#include <iostream>
#include <math.h>

#include "saxpy.h"

static void saxpy(size_t n, real_t a, real_t *x, real_t *y)
{
  for (size_t i = 0; i < n; ++i) {
      y[i] += a * x[i];
  }
}

int main(void)
{
    real_t *x = new real_t[N];
    real_t *y = new real_t[N];

    std::cout << "N: " << N << std::endl;

    for (size_t i = 0; i < N; i++) {
	    x[i] = XVAL;
	    y[i] = YVAL;
    }

    saxpy_timer t;

    saxpy(N, AVAL, x, y);

    double elapsed = t.elapsed_msec();

    saxpy_verify(y);
    std::cout << "Elapsed: " << elapsed << " ms" << std::endl;

    delete [] x;
    delete [] y;

    return 0;
}
