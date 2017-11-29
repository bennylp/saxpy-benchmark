import time

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import saxpy

print("Using device {}".format(cuda.Context.get_device().name()))
print("N: {}".format(saxpy.N))

host_x = np.zeros([saxpy.N], dtype=np.float32) + np.float32(saxpy.XVAL)
host_y = np.zeros([saxpy.N], dtype=np.float32) + np.float32(saxpy.YVAL)
AVAL = np.float32(saxpy.AVAL)

dev_x = cuda.mem_alloc(host_x.nbytes)
dev_y = cuda.mem_alloc(host_y.nbytes)

cuda.memcpy_htod(dev_x, host_x)
cuda.memcpy_htod(dev_y, host_y)

saxpy_mod = SourceModule("""
__global__ void saxpy(size_t n, float a, float *x, float *y)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = index; i < n; i += stride) {
            y[i] += a * x[i];
    }
}
""")

grid = (4096, 1, 1)
block = (256, 1, 1)
saxpy_func = saxpy_mod.get_function("saxpy")
saxpy_func.prepare("NfPP")

t0 = time.time()
saxpy_func.prepared_call(grid, block, saxpy.N, AVAL, dev_x, dev_y)
cuda.Context.synchronize()
elapsed = time.time() - t0
print("Elapsed: {} ms".format(elapsed * 1000))

cuda.memcpy_dtoh(host_y, dev_y)

saxpy.verify(host_y)
