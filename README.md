This contains different implementations of SAXPY (Single Precision A * X Plus Y)
across many backends such as:
 - naive Python loop
 - Python Numpy
 - naive C++ loop
 - C++ CUDA (GPU)
 - OpenCL (both CPU and GPU) 

The implemented SAXPY is basically a single iteration of this loop:

```
  for i=0 to N:
     Y[i] += alpha * X[i]
```

We only measure the time to perform the above loop, and not other things such as 
initialization and data transfers between CPU and GPU.


# Naive Python Loop

This is a naive Python implementation using loop, something like:
```python
  for i in range(N):
      y[i] += AVAL * x[i]
```

Run [saxpy_loop.py](saxpy_loop.py) to see the result.

# Python Numpy

Improved Python version with vectorized Numpy implementation:
```python
  y += AVAL * x
```

Run [saxpy_numpy.py](saxpy_numpy.py) to see the result. You will need Numpy of course. 


# General C++

The C++ codes have been tested on Windows and Mac OS. I assume the instrunctions will be
similar for Linux, but YMMV.

The general requirement is a C++-11 compliant compiler.

The "build system" is basically just a [Makefile](Makefile) which can be executed 
by both `make` or Visual Studio's `nmake`. You'll need to edit the [Makefile](Makefile) manually
to configure things, just follow the instructions in the Makefile.

Once done, run `make` or `nmake` to build the executables.

## Naive C/C++ Loop

This is naive C/C++ loop using CPU:
```c++
static void saxpy(size_t n, real_t a, real_t *x, real_t *y)
{
  for (size_t i = 0; i < n; ++i) {
      y[i] += a * x[i];
  }
}
```
See [saxpy_cpu.cpp](saxpy_cpu.cpp) for the implemention.


# CUDA

CUDA® is a parallel computing platform and programming model developed by NVIDIA for 
(NVIDIA) general computing on graphical processing units (GPGPU). Because NVIDIA is
the market leader in GPU, that makes CUDA the leading API for GPGPU as well and AFAIK
it is the API that is used by pretty much all higher level libraries utilizing GPGPU 
such as TensorFlow and what not.

One of the good thing about CUDA is, if you have a computer from 2008 onwards with an
NVidia GPU on it (such as GeForce 9800 GTX+ or GeForce 9800 GTX+), chances are you can
use CUDA on it.  

The other good things are it has good API (subjective of course), plenty of documentations,
and good community and support.

These are select resources on CUDA:
 - [NVidia CUDA Zone](https://developer.nvidia.com/cuda-zone)
 - [CUDA (Wikipedia)](https://en.wikipedia.org/wiki/CUDA)
 - [Supported GPU (GeForce only)](https://www.geforce.com/hardware/technology/cuda/supported-gpus)
 - [An Even Easier Introduction to CUDA](https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/)


### Building and Running

Requirements: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Make sure `nvcc` and
`nvprof` are available in the PATH.

Configuring: edit the [Makefile](Makefile) and include `saxpy_cuda` in `TARGET` variable.

Building: just run `make` or `nmake`

Source file: [saxpy_cuda.cu](saxpy_cuda.cu)

Running: run `nvprof saxpy_cuda`, and you will see the time it takes to execute saxpy in the
output similar to this:
```
==2168== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
100.00%  7.9757ms         1  7.9757ms  7.9757ms  7.9757ms  saxpy(unsi...
```


# OpenCL

