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
    real_t *host_x = new float[N], *host_y = new float[N];
    real_t *dev_x, *dev_y;

    // initialize x and y arrays on the host
    for (size_t i = 0; i < N; i++) {
        host_x[i] = XVAL;
        host_y[i] = YVAL;
    }

    cudaMalloc((void**) &dev_x, N*sizeof(real_t));
    cudaMalloc((void**) &dev_y, N*sizeof(real_t));

    cudaMemcpy(dev_x, host_x, N*sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, host_y, N*sizeof(real_t), cudaMemcpyHostToDevice);
    
    // Run kernel on 1M elements on the CPU
    int blockSize = 256;
    int numBlocks = 4096;
    saxpy_timer t;
    
    saxpy<<<numBlocks, blockSize>>>(N, AVAL, dev_x, dev_y);

    cudaDeviceSynchronize();
    
    double elapsed = t.elapsed_msec();
    
    // Check for errors (all values should be 3.0f)
    std::cout << "N: " << N << std::endl;
    std::cout << "Total elapsed: " << elapsed << " ms" << std::endl;
    
    cudaMemcpy(host_y, dev_y, N*sizeof(real_t), cudaMemcpyDeviceToHost);
    saxpy_verify(host_y);
    
    // Free memory
    cudaFree(dev_x);
    cudaFree(dev_y);
    delete [] host_x;
    delete [] host_y;
    
    return 0;
}
