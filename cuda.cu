#include <iostream>
#include "saxpy.h"

__global__ void saxpy(size_t n, real_t a, real_t *x, real_t *y)
{
	const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t stride = blockDim.x * gridDim.x;
	
	for (size_t i = index; i < n; i += stride) {
		y[i] += a * x[i];
	}
}

int main(void)
{
	real_t *x, *y;

	cudaMallocManaged(&x, N * sizeof(real_t));
	cudaMallocManaged(&y, N * sizeof(real_t));
  
	// initialize x and y arrays on the host
	for (size_t i = 0; i < N; i++) {
		x[i] = XVAL;
		y[i] = YVAL;
	}

	// Run kernel on 1M elements on the CPU
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	saxpy_timer t;
	
	saxpy<<<numBlocks, blockSize>>>(N, AVAL, x, y);

	cudaDeviceSynchronize();
	
	// Check for errors (all values should be 3.0f)
	saxpy_verify(y);
	std::cout << "Total elapsed: " << t.elapsed_msec() << " ms" << std::endl;
	
	// Free memory
	cudaFree(x);
	cudaFree(y);
	
	return 0;
}
