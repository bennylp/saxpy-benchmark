# SAXPY CPU and GPGPU Benchmarks

**Table of Contents**:

- [Benchmarks](#benchmarks)
- [Results](#results)
   - [Python: Loop vs Numpy (CPU)](#python-loop-vs-numpy-cpu)
   - [Python: Loop vs Numpy 2 (CPU)](#python-loop-vs-numpy-2-cpu)
   - [R: Loop vs Vectorized (CPU)](#r-loop-vs-vectorized-cpu)
   - [Python: Loop vs Numpy vs Pandas (CPU)](#python-loop-vs-numpy-vs-pandas-cpu)
   - [Julia: Loop vs Vector (CPU)](#julia-loop-vs-vector-cpu)
   - [Numpy vs Octave vs R vs Java vs Julia vs C++ (CPU)](#numpy-vs-octave-vs-r-vs-java-vs-julia-vs-c-cpu)
   - [Python Vectorization: Numpy vs Deep Learning Frameworks (CPU)](#python-vectorization-numpy-vs-deep-learning-frameworks-cpu)
   - [Numpy vs Deep Learning Frameworks (GPU and CPU)](#numpy-vs-deep-learning-frameworks-gpu-and-cpu)
   - [Deep Learning Frameworks GPU vs Loop CPU](#deep-learning-frameworks-gpu-vs-loop-cpu)
   - [C++ Parallel APIs (CPU)](#c-parallel-apis-cpu)
   - [C++ GPU (vs CPU)](#c-gpu-vs-cpu)
   - [OpenCL vs PyOpenCL (CPU & GPU)](#opencl-vs-pyopencl-cpu--gpu)
   - [PyCUDA vs C++ (GPU)](#pycuda-vs-c-gpu)
   - [Tensorflow: Python vs C++ (GPU)](#tensorflow-python-vs-c-gpu)
   - [GPU Conclusion](#gpu-conclusion)
   - [Linux Conclusion](#linux-conclusion)
   - [Windows Conclusion](#windows-conclusion)
   - [Conclusion](#conclusion)
- [Machine Specifications](#machine-specifications)
   - [Ubuntu 16.04, NVidia GTX 1080](#ubuntu-1604-nvidia-gtx-1080)
   - [Windows 10, NVidia GTX 1080](#windows-10-nvidia-gtx-1080)


# Benchmarks

The following benchmarks are implemented:

- **PyCUDA [gpu]** ([src/saxpy_pycuda.py](src/saxpy_pycuda.py))
  Implementation with [PyCUDA](https://mathema.tician.de/software/pycuda/), the Python wrapper for [CUDA](https://developer.nvidia.com/cuda-toolkit).

- **Py CNTK [gpu]** ([src/saxpy_cntk.py](src/saxpy_cntk.py))
  Implementation for CPU and GPU with [CNTK](https://cntk.ai/), a deep learning library.

- **R (loop) [cpu]** ([src/saxpy_loop.R](src/saxpy_loop.R))
  Simple loop in [R](https://www.r-project.org/), a free software environment for statistical computing and graphics.

- **R (data.table) [cpu]** ([src/saxpy_datatable.R](src/saxpy_datatable.R))
  Implementation with `data.table` in [R](https://www.r-project.org/), a free software environment for statistical computing and graphics.

- **Py CNTK [cpu]** ([src/saxpy_cntk.py](src/saxpy_cntk.py))
  Implementation for CPU and GPU with [CNTK](https://cntk.ai/), a deep learning library.

- **Julia (loop) [cpu]** ([src/saxpy_loop.jl](src/saxpy_loop.jl))
  Plain loop in [Julia](https://julialang.org/) programming language.

- **C++ cuBLAS [gpu]** ([src/saxpy_cublas.cpp](src/saxpy_cublas.cpp))
  A GPU implementation with NVidia [cuBLAS](https://developer.nvidia.com/cublas), a fast GPU-accelerated implementation of the standard basic linear algebra subroutines (BLAS).

- **Py Numpy [cpu]** ([src/saxpy_numpy.py](src/saxpy_numpy.py))
  Vectorized implementation with Python [Numpy](http://www.numpy.org/) array.

- **Octave [cpu]** ([src/saxpy.m](src/saxpy.m))
  Implementation in [GNU Octave](https://www.gnu.org/software/octave/), a high-level language primarily intended for numerical computations.

- **Java loop [cpu]** ([src/SaxpyLoop.java](src/SaxpyLoop.java))
  Plain Java loop

- **R (data.frame) [cpu]** ([src/saxpy_dataframe.R](src/saxpy_dataframe.R))
  Implementation with `data.frame` in [R](https://www.r-project.org/), a free software environment for statistical computing and graphics.

- **C++ OCL [gpu]** ([src/saxpy_ocl1.cpp](src/saxpy_ocl1.cpp))
  Parallel programming with [OpenCL](https://en.wikipedia.org/wiki/OpenCL), a framework for writing programs that execute across heterogeneous platforms consisting of central processing units (CPUs), graphics processing units (GPUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs) and other processors or hardware accelerators.

- **C++ TensorFlow [gpu]** ([src/saxpy_tf.cc](src/saxpy_tf.cc))
  Implementation in C++ for GPU with [TensorFlow](https://www.tensorflow.org/), a deep learning library.

- **R (matrix) [cpu]** ([src/saxpy_matrix.R](src/saxpy_matrix.R))
  Implementation with matrix in [R](https://www.r-project.org/), a free software environment for statistical computing and graphics.

- **Py MXNet [cpu]** ([src/saxpy_mxnet.py](src/saxpy_mxnet.py))
  Implementation for CPU and GPU with [MXNet](https://mxnet.incubator.apache.org/), a deep learning library.

- **Py TensorFlow [gpu]** ([src/saxpy_tf.py](src/saxpy_tf.py))
  Implementation for CPU and GPU with [TensorFlow](https://www.tensorflow.org/), a deep learning library.

- **C++ loop [cpu]** ([src/saxpy_cpu.cpp](src/saxpy_cpu.cpp))
  Plain C++ `for` loop

- **Julia (vec) [cpu]** ([src/saxpy_array.jl](src/saxpy_array.jl))
  Vectorized implementation with array in [Julia](https://julialang.org/) programming language.

- **R (array) [cpu]** ([src/saxpy_array.R](src/saxpy_array.R))
  Implementation with array in [R](https://www.r-project.org/), a free software environment for statistical computing and graphics.

- **Py TensorFlow [cpu]** ([src/saxpy_tf.py](src/saxpy_tf.py))
  Implementation for CPU and GPU with [TensorFlow](https://www.tensorflow.org/), a deep learning library.

- **Python loop [cpu]** ([src/saxpy_loop.py](src/saxpy_loop.py))
  Simple Python `for` loop.

- **Py Pandas [cpu]** ([src/saxpy_pandas.py](src/saxpy_pandas.py))
  Vectorized implementation with Python [Pandas](https://pandas.pydata.org/) dataframe.

- **C++ CUDA [gpu]** ([src/saxpy_cuda.cpp](src/saxpy_cuda.cpp))
  Low level implementation with the base NVidia [CUDA](https://developer.nvidia.com/cuda-toolkit) toolkit.

- **C++ Thrust [gpu]** ([src/saxpy_trust.cpp](src/saxpy_trust.cpp))
  A GPU implementation with NVidia [Thrust](https://thrust.github.io/), a parallel algorithms library which resembles the C++ Standard Template Library (STL). Thrust is included with [CUDA](https://developer.nvidia.com/cuda-toolkit) toolkit.

- **Py MXNet [gpu]** ([src/saxpy_mxnet.py](src/saxpy_mxnet.py))
  Implementation for CPU and GPU with [MXNet](https://mxnet.incubator.apache.org/), a deep learning library.

- **C++ OMP [cpu]** ([src/saxpy_omp.cpp](src/saxpy_omp.cpp))
  Parallel programming with [OpenMP](http://www.openmp.org/). Only CPU version is implemented.

- **C++ Bulk [gpu]** ([src/saxpy_bulk.cpp](src/saxpy_bulk.cpp))
  A GPU implementation with [Bulk](https://github.com/jaredhoberock/bulk), yet another parallel algorithms on top of CUDA.

- **PyOCL [cpu]** ([src/saxpy_pyocl.py](src/saxpy_pyocl.py))
  CPU and GPU implementation with [PyOpenCL](https://mathema.tician.de/software/pyopencl/), the Python wrapper for [OpenCL](https://en.wikipedia.org/wiki/OpenCL).

- **PyOCL [gpu]** ([src/saxpy_pyocl.py](src/saxpy_pyocl.py))
  CPU and GPU implementation with [PyOpenCL](https://mathema.tician.de/software/pyopencl/), the Python wrapper for [OpenCL](https://en.wikipedia.org/wiki/OpenCL).

- **C++ OCL [cpu]** ([src/saxpy_ocl1.cpp](src/saxpy_ocl1.cpp))
  Parallel programming with [OpenCL](https://en.wikipedia.org/wiki/OpenCL), a framework for writing programs that execute across heterogeneous platforms consisting of central processing units (CPUs), graphics processing units (GPUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs) and other processors or hardware accelerators.


# Results

## Python: Loop vs Numpy (CPU)

Comparison between simple Python loop and Numpy

- Py Numpy [cpu] ([src/saxpy_numpy.py](src/saxpy_numpy.py))
- Python loop [cpu] ([src/saxpy_loop.py](src/saxpy_loop.py))

![results/charts-en/python-loop-vs-numpy-linux-cpu.png](results/charts-en/python-loop-vs-numpy-linux-cpu.png?raw=true "results/charts-en/python-loop-vs-numpy-linux-cpu.png")

## Python: Loop vs Numpy 2 (CPU)

Same as above, on both Linux and Windows

- Py Numpy [cpu] ([src/saxpy_numpy.py](src/saxpy_numpy.py))
- Python loop [cpu] ([src/saxpy_loop.py](src/saxpy_loop.py))

![results/charts-en/python-loop-vs-numpy-cpu.png](results/charts-en/python-loop-vs-numpy-cpu.png?raw=true "results/charts-en/python-loop-vs-numpy-cpu.png")

## R: Loop vs Vectorized (CPU)

Benchmarking various vectorization methods in R (array, matrix, data.frame, data.table) vs plain loop

- R (array) [cpu] ([src/saxpy_array.R](src/saxpy_array.R))
- R (data.frame) [cpu] ([src/saxpy_dataframe.R](src/saxpy_dataframe.R))
- R (data.table) [cpu] ([src/saxpy_datatable.R](src/saxpy_datatable.R))
- R (loop) [cpu] ([src/saxpy_loop.R](src/saxpy_loop.R))
- R (matrix) [cpu] ([src/saxpy_matrix.R](src/saxpy_matrix.R))

![results/charts-en/r-loop-vs-vec.png](results/charts-en/r-loop-vs-vec.png?raw=true "results/charts-en/r-loop-vs-vec.png")

## Python: Loop vs Numpy vs Pandas (CPU)

Benchmarking the performance of Numpy vs Panda (vs plain Python loop)

- Py Numpy [cpu] ([src/saxpy_numpy.py](src/saxpy_numpy.py))
- Py Pandas [cpu] ([src/saxpy_pandas.py](src/saxpy_pandas.py))
- Python loop [cpu] ([src/saxpy_loop.py](src/saxpy_loop.py))

![results/charts-en/python-loop-vs-numpy-vs-pandas-cpu.png](results/charts-en/python-loop-vs-numpy-vs-pandas-cpu.png?raw=true "results/charts-en/python-loop-vs-numpy-vs-pandas-cpu.png")

## Julia: Loop vs Vector (CPU)

Comparing the performance of Julia loop vs Julia vector/array (vs C++)

- C++ loop [cpu] ([src/saxpy_cpu.cpp](src/saxpy_cpu.cpp))
- Julia (loop) [cpu] ([src/saxpy_loop.jl](src/saxpy_loop.jl))
- Julia (vec) [cpu] ([src/saxpy_array.jl](src/saxpy_array.jl))

![results/charts-en/julia-loop-vs-vector.png](results/charts-en/julia-loop-vs-vector.png?raw=true "results/charts-en/julia-loop-vs-vector.png")

## Numpy vs Octave vs R vs Java vs Julia vs C++ (CPU)

Comparing the performance of SAXPY in different programming languages

- C++ loop [cpu] ([src/saxpy_cpu.cpp](src/saxpy_cpu.cpp))
- Java loop [cpu] ([src/SaxpyLoop.java](src/SaxpyLoop.java))
- Julia (loop) [cpu] ([src/saxpy_loop.jl](src/saxpy_loop.jl))
- Julia (vec) [cpu] ([src/saxpy_array.jl](src/saxpy_array.jl))
- Octave [cpu] ([src/saxpy.m](src/saxpy.m))
- Py Numpy [cpu] ([src/saxpy_numpy.py](src/saxpy_numpy.py))
- R (array) [cpu] ([src/saxpy_array.R](src/saxpy_array.R))

![results/charts-en/script-vs-script-vs-java-vs-c++-cpu.png](results/charts-en/script-vs-script-vs-java-vs-c++-cpu.png?raw=true "results/charts-en/script-vs-script-vs-java-vs-c++-cpu.png")

## Python Vectorization: Numpy vs Deep Learning Frameworks (CPU)

SAXPY array operation in Numpy vs machine learning frameworks such as Tensorflow, MXNet, and CNTK. Only tested on Linux.

Note: CNTK result is way off, not sure why. Please have a look at the source code, maybe I did something wrong.

- Py CNTK [cpu] ([src/saxpy_cntk.py](src/saxpy_cntk.py))
- Py MXNet [cpu] ([src/saxpy_mxnet.py](src/saxpy_mxnet.py))
- Py Numpy [cpu] ([src/saxpy_numpy.py](src/saxpy_numpy.py))
- Py TensorFlow [cpu] ([src/saxpy_tf.py](src/saxpy_tf.py))

![results/charts-en/vectorized-numpy-vs-frameworks-cpu.png](results/charts-en/vectorized-numpy-vs-frameworks-cpu.png?raw=true "results/charts-en/vectorized-numpy-vs-frameworks-cpu.png")

## Numpy vs Deep Learning Frameworks (GPU and CPU)

Same as above, but on GPU as well

- Py CNTK [cpu] ([src/saxpy_cntk.py](src/saxpy_cntk.py))
- Py CNTK [gpu] ([src/saxpy_cntk.py](src/saxpy_cntk.py))
- Py MXNet [cpu] ([src/saxpy_mxnet.py](src/saxpy_mxnet.py))
- Py MXNet [gpu] ([src/saxpy_mxnet.py](src/saxpy_mxnet.py))
- Py Numpy [cpu] ([src/saxpy_numpy.py](src/saxpy_numpy.py))
- Py TensorFlow [cpu] ([src/saxpy_tf.py](src/saxpy_tf.py))
- Py TensorFlow [gpu] ([src/saxpy_tf.py](src/saxpy_tf.py))

![results/charts-en/vectorized-numpy-vs-frameworks-gpu.png](results/charts-en/vectorized-numpy-vs-frameworks-gpu.png?raw=true "results/charts-en/vectorized-numpy-vs-frameworks-gpu.png")

## Deep Learning Frameworks GPU vs Loop CPU

Comparing frameworks running on GPU with naive C++ loop running on CPU.

- C++ loop [cpu] ([src/saxpy_cpu.cpp](src/saxpy_cpu.cpp))
- Py CNTK [gpu] ([src/saxpy_cntk.py](src/saxpy_cntk.py))
- Py MXNet [gpu] ([src/saxpy_mxnet.py](src/saxpy_mxnet.py))
- Py TensorFlow [gpu] ([src/saxpy_tf.py](src/saxpy_tf.py))

![results/charts-en/frameworks-gpu-vs-c++-cpu.png](results/charts-en/frameworks-gpu-vs-c++-cpu.png?raw=true "results/charts-en/frameworks-gpu-vs-c++-cpu.png")

## C++ Parallel APIs (CPU)

Comparing naive C++ loop with several parallel programming APIs (OpenCL and OpenMP) on CPU.

- C++ OCL [cpu] ([src/saxpy_ocl1.cpp](src/saxpy_ocl1.cpp))
- C++ OMP [cpu] ([src/saxpy_omp.cpp](src/saxpy_omp.cpp))
- C++ loop [cpu] ([src/saxpy_cpu.cpp](src/saxpy_cpu.cpp))

![results/charts-en/parallel-c++-cpu.png](results/charts-en/parallel-c++-cpu.png?raw=true "results/charts-en/parallel-c++-cpu.png")

## C++ GPU (vs CPU)

Comparing various C++ GPU libraries (CUDA, OpenCL, Thrust, Bulk, cuBLAS)

- C++ Bulk [gpu] ([src/saxpy_bulk.cpp](src/saxpy_bulk.cpp))
- C++ CUDA [gpu] ([src/saxpy_cuda.cpp](src/saxpy_cuda.cpp))
- C++ OCL [gpu] ([src/saxpy_ocl1.cpp](src/saxpy_ocl1.cpp))
- C++ Thrust [gpu] ([src/saxpy_trust.cpp](src/saxpy_trust.cpp))
- C++ cuBLAS [gpu] ([src/saxpy_cublas.cpp](src/saxpy_cublas.cpp))
- C++ loop [cpu] ([src/saxpy_cpu.cpp](src/saxpy_cpu.cpp))

![results/charts-en/c++-cpu-vs-gpu.png](results/charts-en/c++-cpu-vs-gpu.png?raw=true "results/charts-en/c++-cpu-vs-gpu.png")

## OpenCL vs PyOpenCL (CPU & GPU)

Comparing C++ OpenCL with PyOpenCL, the OpenCL Python wrapper.

- C++ OCL [cpu] ([src/saxpy_ocl1.cpp](src/saxpy_ocl1.cpp))
- C++ OCL [gpu] ([src/saxpy_ocl1.cpp](src/saxpy_ocl1.cpp))
- PyOCL [cpu] ([src/saxpy_pyocl.py](src/saxpy_pyocl.py))
- PyOCL [gpu] ([src/saxpy_pyocl.py](src/saxpy_pyocl.py))

![results/charts-en/pyopencl-vs-opencl.png](results/charts-en/pyopencl-vs-opencl.png?raw=true "results/charts-en/pyopencl-vs-opencl.png")

## PyCUDA vs C++ (GPU)

Comparing PyCUDA (Python CUDA wrapper) with native C++ CUDA GPU

- C++ CUDA [gpu] ([src/saxpy_cuda.cpp](src/saxpy_cuda.cpp))
- PyCUDA [gpu] ([src/saxpy_pycuda.py](src/saxpy_pycuda.py))

![results/charts-en/pycuda-vs-c++.png](results/charts-en/pycuda-vs-c++.png?raw=true "results/charts-en/pycuda-vs-c++.png")

## Tensorflow: Python vs C++ (GPU)

Comparing Tensorflow C++ and Python performance

- C++ TensorFlow [gpu] ([src/saxpy_tf.cc](src/saxpy_tf.cc))
- Py TensorFlow [gpu] ([src/saxpy_tf.py](src/saxpy_tf.py))

![results/charts-en/tensorflow-python-vs-c++.png](results/charts-en/tensorflow-python-vs-c++.png?raw=true "results/charts-en/tensorflow-python-vs-c++.png")

## GPU Conclusion

Benchmarking various GPU APIs (only on Linux since it has the most APIs)

**Excluded** from this chart:

![results/charts-en/conclusion-gpus.png](results/charts-en/conclusion-gpus.png?raw=true "results/charts-en/conclusion-gpus.png")

## Linux Conclusion

**Excluded** from this chart:
- Python loop [cpu] ([src/saxpy_loop.py](src/saxpy_loop.py))
- R (loop) [cpu] ([src/saxpy_loop.R](src/saxpy_loop.R))

![results/charts-en/conclusion-linux.png](results/charts-en/conclusion-linux.png?raw=true "results/charts-en/conclusion-linux.png")

## Windows Conclusion

**Excluded** from this chart:
- Python loop [cpu] ([src/saxpy_loop.py](src/saxpy_loop.py))
- R (loop) [cpu] ([src/saxpy_loop.R](src/saxpy_loop.R))
- C++ TensorFlow [gpu] ([src/saxpy_tf.cc](src/saxpy_tf.cc))
- Py CNTK [gpu] ([src/saxpy_cntk.py](src/saxpy_cntk.py))
- Py CNTK [cpu] ([src/saxpy_cntk.py](src/saxpy_cntk.py))

![results/charts-en/conclusion-windows.png](results/charts-en/conclusion-windows.png?raw=true "results/charts-en/conclusion-windows.png")

## Conclusion

**Excluded** from this chart:
- Python loop [cpu] ([src/saxpy_loop.py](src/saxpy_loop.py))
- R (loop) [cpu] ([src/saxpy_loop.R](src/saxpy_loop.R))
- C++ TensorFlow [gpu] ([src/saxpy_tf.cc](src/saxpy_tf.cc))
- Py CNTK [gpu] ([src/saxpy_cntk.py](src/saxpy_cntk.py))
- Py CNTK [cpu] ([src/saxpy_cntk.py](src/saxpy_cntk.py))

![results/charts-en/conclusion.png](results/charts-en/conclusion.png?raw=true "results/charts-en/conclusion.png")



# Machine Specifications
## Ubuntu 16.04, NVidia GTX 1080

Note: same machine as Windows below (dual-boot)

|    |    |
|----|----|
| System | Intel i7-6700 CPU @ 3.40GHz 16GB RAM 4x2 cores (HT) |
| OS | Ubuntu Linux 16.04 64bit |
| GPU | NVidia GeForce GTX 1080 8GB |
| C++ Compiler | g++ 5.4.0 |
| Python3 | 3.5.2 64bit |
| TensorFlow | TensorFlow 1.4 (GPU) |
| CUDA | CUDA 9.0.61 |
|  | CudNN7 |
| OpenCL | - Khronos OpenCL header 1.2 |
|  | - Intel OpenCL driver 16.1.1 |
|  | - NVidia OpenCL 1.2 driver |
| PyOpenCL | version 2015.1 |
| Octave | version 4.0.0 64bit |
| R | version 3.2.3 64bit |
| MXNet | mxnet-cu90 (0.12.1) |
| CNTK | CNTK 2.3.1 (CUDA-8, CudNN6) |

## Windows 10, NVidia GTX 1080

Note: same machine as Linux above (dual-boot)

|    |    |
|----|----|
| System | Intel i7-6700 CPU @ 3.40GHz 16GB RAM 4x2 cores (HT) |
| OS | Windows 10 64bit |
| GPU | NVidia GeForce GTX 1080 8GB |
| C++ Compiler | Visual Studio 2015 C++ compiler 64bit version |
| Python | 2.7.12 64bit |
| Python3 | 3.5.3 64bit |
| TensorFlow | TensorFlow 1.4 (GPU) |
| CUDA | Version 8.0.61 |
| OpenCL | - Intel OpenCL SDK Version 7.0.0.2519 |
|  | - OpenCL from CUDA SDK |
| PyOpenCL | version 2017.2 |
| Octave | version 4.2.1 64bit |
| R | version 3.4.2 64bit |


