#include <iostream>
#include <cublas_v2.h>
#include "saxpy.h"

int main()
{
#define CE(op) if ((status = op) != CUBLAS_STATUS_SUCCESS) { std::cerr << "Error: " #op << " [status=" << status << "]\n"; return 1; }
   cublasStatus_t status;
   cublasHandle_t h = nullptr;
   float *host_x, *host_y;

   std::cout << "N: " << N << std::endl;

   host_x = new float[N];
   host_y = new float[N];

   //cublasInit();
   CE( cublasCreate(&h) );

   for (int i=0; i<N; ++i) {
      host_x[i] = XVAL;
      host_y[i] = YVAL;
   }
   float *dev_x, *dev_y;
   cudaMalloc( (void**)&dev_x, N*sizeof(float));
   cudaMalloc( (void**)&dev_y, N*sizeof(float));

   CE( cublasSetVector(N, sizeof(host_x[0]), host_x, 1, dev_x, 1) );
   CE( cublasSetVector(N, sizeof(host_y[0]), host_y, 1, dev_y, 1) );

   cudaDeviceSynchronize();

   saxpy_timer t;
   cublasSaxpy(h, N, &AVAL, dev_x, 1, dev_y, 1);
   cudaDeviceSynchronize();
   double elapsed = t.elapsed_msec();

   std::cout << "Elapsed: " << elapsed << " ms\n";

   cublasGetVector(N, sizeof(host_y[0]), dev_y, 1, host_y, 1);
   saxpy_verify(host_y);

   if (h)
      cublasDestroy(h);
   cudaFree(dev_y);
   cudaFree(dev_x);
   delete [] host_y;
   delete [] host_x;

   return 0;
}
