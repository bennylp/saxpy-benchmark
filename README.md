# SAXPY GPGPU Benchmark

SAXPY (Single Precision A * X Plus Y) is basically:
```python
  for i=0 to N:
     Y[i] = A * X[i] + Y[i]
```

This repository contains several implementations of SAXPY such as:
 - naive Python loop [[saxpy_loop.py](saxpy_loop.py)]
 - Python Numpy [[saxpy_numpy.py](saxpy_numpy.py)]
 - naive C++ loop [[saxpy_cpu.cpp](saxpy_cpu.cpp)]
 - C++ CUDA (GPU) [[saxpy_cuda.cu](saxpy_cuda.cu)]
 - OpenCL (CPU and GPU) [[saxpy_ocl1.cpp](saxpy_ocl1.cpp)] 
 - PyOpenCL (CPU and GPU) [[saxpy_pyocl.py](saxpy_pyocl.py)]

For the benchmark, we only measure the time to perform the actual loop and not other things 
such as initialization and data transfers between CPU and GPU, which most likely will exceed
the loop time since our loop is very simple.


# Results

Here are some sample results.

## i7-6700, Windows10, GeForce GTX 745 GPU
 
- Hardware:
  - Intel i7-6700 CPU @ 3.40GHz (4 cores, HT capable)
  - NVidia GeForce GTX 745 graphics card
- Software:
  - Windows 10 64bit
  - Visual Studio 2015 C++ compiler 64bit version
  - [CUDA Version 8.0.61](https://developer.nvidia.com/cuda-zone)
  - [Intel OpenCL SDK Version 7.0.0.2519](https://software.intel.com/en-us/intel-opencl) 
  - Python 2.7.12 64bit
  - [PyOpenCL version 2017.2](https://mathema.tician.de/software/pyopencl/)
- Setup:
  - N is 2^26, or about 67 million elements.

## MacBook Pro 13" Late 2013, onboard Intel Iris GPU



# Naive Python Loop

This is a naive Python implementation using loop, something like:
```python
  x = [XVAL] * N
  y = [YVAL] * N

  for i in range(N):
      y[i] += AVAL * x[i]
```

Run [saxpy_loop.py](saxpy_loop.py) to see the result. On my computer, it takes this much:
```
N: 67108864
Elapsed: 11628.000021 ms
Error: 0.0
```

# Python Numpy

Improved Python version with vectorized Numpy implementation:
```python
  x = np.ones([N]) * XVAL
  y = np.ones([N]) * YVAL

  y += AVAL * x
```

Run [saxpy_numpy.py](saxpy_numpy.py) to see the result. You will need Numpy of course. Here is
the output on my computer:
```
N: 67108864
Elapsed: 262.000083923 ms
Error: 0.0
```

It shows about 44x speed up over the plain Python loop version. Now let's see how things go with C++.


# General C++ Instructions

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
See [saxpy_cpu.cpp](saxpy_cpu.cpp) for the implemention. This file is always enabled in the
Makefile, and there should be nothing to configure. Just run `make` to get it built. 

Running `saxpy_cpu` on my computer gives the following output
```
N: 67108864
Errors: 0
Elapsed: 41.6829 ms
```

So even the simplest C/C++ version is over 6x faster than Numpy.


# CUDA

CUDA� is a parallel computing platform and programming model developed by NVIDIA for 
general computing on (NVIDIA) graphical processing units (GPGPU). Because NVIDIA is
THE undisputed leader in GPU/GPGPU market, that makes CUDA the leading API for GPGPU area. 
In machine learning, AFAIK it is the API that is used by pretty much all ML frameworks utilizing
GPGPU such as TensorFlow, MXNet, and what not.

One of the good thing about CUDA is, if you have a computer from 2008 onwards with an
NVidia GPU on it (say GeForce 9800 GTX+ or GeForce G102M), chances are you can
use CUDA on it. 

The other good things are it has good API (subjective, of course), plenty of documentations,
and good community and forum support.

Some select resources on CUDA:
 - [NVidia CUDA Zone](https://developer.nvidia.com/cuda-zone)
 - [CUDA (Wikipedia)](https://en.wikipedia.org/wiki/CUDA)
 - [Supported GPU (GeForce only)](https://www.geforce.com/hardware/technology/cuda/supported-gpus)
 - [An Even Easier Introduction to CUDA](https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/)

### Installation

1. Make sure you have CUDA capable graphic cards (see Supported GPU link above)
2. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). 
3. Make sure `nvcc` is available in the PATH of your command prompt. 

### Implementation

It's implemented in [saxpy_cuda.cu](saxpy_cuda.cu) file.

### Configuring and Building

Edit the [Makefile](Makefile) and include `saxpy_cuda` in `TARGET` variable.

Configure the flags in Step 3 (see the [Makefile](Makefile)) to match your environment.

Then run `make` (or `nmake`) to build things.

### Running

Run `saxpy_cuda`.

On my computer:
```
N: 67108864
Total elapsed: 32.2043 ms
Errors: 0
```

It's not so bad for a consumer grade graphics card.

# OpenCL

Open Computing Language (OpenCL) is a framework for writing parallel programs in CPUs, GPUs,
DSPs, FPGAs, and other processors or hardware accelerators. Unlike CUDA, OpenCL is available 
across much wider range of platforms, including NVidia hardware as well, and implementations
are available from AMD, Apple, ARM, Creative, IBM, Intel, Nvidia, Samsung, etc. 

It's even installed by default on MacOS since MacOS 10.7. So this sounds like a good framework 
to try.

Some select resources on OpenCL:
- Beginner:
  - [OpenCL (Wikipedia)](https://en.wikipedia.org/wiki/OpenCL)
  - [A Gentle Introduction to OpenCL (Dr.Dobb's)](http://www.drdobbs.com/parallel/a-gentle-introduction-to-opencl/231002854)
  - [OpenCL on Visual Studio : Configuration tutorial for the confused](https://medium.com/@pratikone/opencl-on-visual-studio-configuration-tutorial-for-the-confused-3ec1c2b5f0ca)
- Intermediate/advanced:
  - [The OpenCL Programming (free ebook)](https://www.fixstars.com/en/opencl/book/OpenCLProgrammingBook/contents/)
  - [OpenCL Best Practices (PDF)](https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/OpenCL_Best_Practices_Guide.pdf)

### Implementations

Two implementations are provided, [saxpy_ocl1.cpp](saxpy_ocl1.cpp) and [saxpy_ocl2.cpp](saxpy_ocl2.cpp).
One is using OpenCL C++ API (provided by `cl.hpp`), and the other is using the C API (`cl.h` or `OpenCL.h`). 
You can see that C++ code is shorter and easier to use than the C one. 
There is no performance difference between the two as far as I can see.


### Installation

OpenCL consists of two things, the **SDK** and the **drivers**.
 
The SDK is basically the `cl.h` or `opencl.h` and its dependency headers, optionally `cl.hpp`
for the C++ API, and the library file (`OpenCL.lib` on Windows).

The drivers are the one that manages the execution of your parallel code (called kernel) in the
target device, which can be CPU, GPU, etc.

If you're on MacOS, OpenCL is already installed by default.

If you have installed NVidia CUDA SDK, actually it includes OpenCL SDK in its standard
`include` and `lib` directories (you can see the locations in the [Makefile](Makefile)). 
This OpenCL SDK is ready to use. But unless you install other
drivers, this will only support NVidia cards. It doesn't support `cpu` target. 

For more comprehensive SDK, you can get [Intel SDK for OpenCL](https://software.intel.com/en-us/intel-opencl),
available for Windows and Linux. On Windows, it provides nice integration with Visual Studio, 
which many will appreciate I'm sure.

For the drivers, the [Intel OpenCL driver](https://software.intel.com/en-us/articles/opencl-drivers)
provides the driver for Intel Core and Xeon processors. I assume drivers for the graphics cards
will be available from the manufacturer's website, or perhaps via Windows Update mechanism if you're on Windows.  


### Configure and Build

Edit the [Makefile](Makefile) and include `saxpy_ocl1` and `saxpy_ocl2` in `TARGET` 
variable. 

Configure the flags in Step 3 (see the [Makefile](Makefile)) to match your environment.

Run `make` (or `nmake`) to build things.

### Running

Running `saxpy_ocl1` or `saxpy_ocl2` without any arguments will run saxpy on default
device (in my case, it's the GPU). 

Here's the output on my computer:
```
C:\...\> saxpy_ocl1
Platform "NVIDIA CUDA ". Devices:
 - [gpu ] NVIDIA Corporation : GeForce GTX 745
   (Max compute units: 3, max work group size: 1024)

Platform "Intel(R) OpenCL ". Devices:
 - [cpu ] Intel(R) Corporation : Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
   (Max compute units: 8, max work group size: 8192)

Platform "Experimental OpenCL 2.1 CPU Only Platform ". Devices:
 - [cpu ] Intel(R) Corporation : Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
   (Max compute units: 8, max work group size: 8192)

Using NVIDIA Corporation  GeForce GTX 745
Elapsed: 31.7932 ms
Errors: 0
```

In this case, the result is more or less the same as the CUDA version.


### Running on CPU

You can tell it to run on GPU or CPU by giving it `cpu` or `gpu` argument. Let's give it
`cpu`.
 
I was expecting that on CPU OpenCL will automagically partition our code to run in parallel
in the most efficient way. Here is the result.

```
C:\..> saxpy_ocl1 cpu
Platform "NVIDIA CUDA ". Devices:
 - [gpu ] NVIDIA Corporation : GeForce GTX 745
   (Max compute units: 3, max work group size: 1024)

Platform "Intel(R) OpenCL ". Devices:
 - [cpu ] Intel(R) Corporation : Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
   (Max compute units: 8, max work group size: 8192)

Platform "Experimental OpenCL 2.1 CPU Only Platform ". Devices:
 - [cpu ] Intel(R) Corporation : Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
   (Max compute units: 8, max work group size: 8192)

Using Intel(R) Corporation  Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
Elapsed: 120.771 ms
Errors: 0
```

So the result is pretty disappointing. I'm not sure why the performance is so slow, about four
times slover. My CPU have four cores though, not sure if it has something to do with it.

# PyOpenCL

PyOpenCL is a binding for OpenCL in Python.  See the documentation in 
[PyOpenCL web page](https://mathema.tician.de/software/pyopencl/) for installation guide etc.
FWIW, I use the prebuilt x64 binary for Python 2.7.

Our SAXPY implementation is [saxpy_pyocl.py](saxpy_pyocl.py).

Here is my test result:
```
C:\..> saxpy_pyocl.py
PyOpenCL version 2017.2
Using GeForce GTX 745
N: 67108864
Elapsed: 221.999883652 ms
Error: 0.0
```

So the result is a long way (almost 7x slower) than the OpenCL C++ version. In fact the result
is only slightly faster than the Numpy version.

### Running on CPU

The [saxpy_pyocl.py](saxpy_pyocl.py) may take `cpu` or `gpu` argument. Giving it `cpu` gives
the following result:
```
C:\..> saxpy_pyocl.py cpu
Using Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
N: 67108864
C:\Anaconda2\lib\site-packages\pyopencl\cffi_cl.py:1502: CompilerWarning: Non-empty compiler output encountered. 
Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
  "to see more.", CompilerWarning)
Elapsed: 46.9999313354 ms
Error: 0.0
```

Interestingly the PyOpenCL CPU result is a lot (more than 4x) faster than the PyOpenCL GPU.
In fact, it is faster than the OpenCL C++ version for CPU!
