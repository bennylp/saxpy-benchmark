#include <iostream>
#include <math.h>

#include "saxpy.h"

static void saxpy(size_t n, real_t a, real_t *x, real_t *y)
{
  for (size_t i = 0; i < n; ++i) {
      y[i] = a * x[i] + y[i];
  }
}

int main(void)
{
	real_t *x = new real_t[N];
	real_t *y = new real_t[N];

	std::cout << "N: " << N << std::endl;

	// initialize x and y arrays on the host
	for (size_t i = 0; i < N; i++) {
		x[i] = XVAL;
		y[i] = YVAL;
	}

	// Run kernel on 1M elements on the CPU
	Timer t;
	saxpy(N, AVAL, x, y);
	double elapsed = t.elapsed();

	// Check for errors (all values should be 3.0f)
	float max_error = 0.0f;
	for (size_t i = 0; i < N; i++)
		max_error = fmax(max_error, fabs(y[i] - (AVAL * XVAL + YVAL)));
	std::cout << "Max error: " << max_error << std::endl;

	std::cout << "Elapsed: " << elapsed * 1000 << " ms" << std::endl;

	delete [] x;
	delete [] y;

	return 0;
}
