# SAXPY CPU and GPGPU Benchmark
## Machine Specifications
### i7-6700 3.40GHz 4 cores CPU, NVidia GTX 1080 GPU

|    |    |
|----|----|
| System | HP Pavilion 550-227 desktop |
|  | Intel i7-6700 CPU @ 3.40GHz 16GB RAM |
| OS | Ubuntu Linux 16.04 64bit |
| GPU | NVidia GeForce GTX 1080 8GB mem |
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

### i7-6700 3.40GHz 4 cores CPU, NVidia GTX 475 GPU

|    |    |
|----|----|
| System | HP Pavilion 550-227 desktop |
|  | Intel i7-6700 CPU @ 3.40GHz (4 cores, HT capable) |
| OS | Windows 10 64bit |
| GPU | NVidia GeForce GTX 475 4GB mem |
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

### MacBook Pro 13" late 2013, on board Intel Iris GPU

|    |    |
|----|----|
| System | MacBook Pro 13" late 2013 |
| OS | OS X 10.11.14 |
| GPU | on board Intel Iris |
| OpenCL | preinstalled (XCode 5.1.1) |
| Python | 3.5.2 |
| TensorFlow | 1.0.1 |
| Octave | 3.8.0 |
| Java | 1.8.0_92 |


## Benchmarks
### Python: Loop vs Numpy (CPU)

Comparison between simple Python loop and Numpy

![results/charts-en/python-loop-vs-numpy-linux-cpu.png](results/charts-en/python-loop-vs-numpy-linux-cpu.png?raw=true "results/charts-en/python-loop-vs-numpy-linux-cpu.png")

### Python: Loop vs Numpy 2 (CPU)

Same as above, on both Linux and Windows

![results/charts-en/python-loop-vs-numpy-cpu.png](results/charts-en/python-loop-vs-numpy-cpu.png?raw=true "results/charts-en/python-loop-vs-numpy-cpu.png")

### R: Loop vs Vectorized (CPU)

Implementation with various methods in R

![results/charts-en/r-loop-vs-vec.png](results/charts-en/r-loop-vs-vec.png?raw=true "results/charts-en/r-loop-vs-vec.png")

### Python: Loop vs Numpy vs Pandas (CPU)

![results/charts-en/python-loop-vs-numpy-vs-pandas-cpu.png](results/charts-en/python-loop-vs-numpy-vs-pandas-cpu.png?raw=true "results/charts-en/python-loop-vs-numpy-vs-pandas-cpu.png")

### Julia: Loop vs Vector (CPU)

![results/charts-en/julia-loop-vs-vector.png](results/charts-en/julia-loop-vs-vector.png?raw=true "results/charts-en/julia-loop-vs-vector.png")

### Numpy vs Octave vs R vs Java vs Julia vs C++ (CPU)

Comparison among different programming languages

![results/charts-en/script-vs-script-vs-java-vs-c++-cpu.png](results/charts-en/script-vs-script-vs-java-vs-c++-cpu.png?raw=true "results/charts-en/script-vs-script-vs-java-vs-c++-cpu.png")

### Python Vectorization: Numpy vs Machine Learning Frameworks (CPU)

SAXPY array operation in Numpy vs machine learning frameworks such as Tensorflow and MXNet

![results/charts-en/vectorized-numpy-vs-frameworks-cpu.png](results/charts-en/vectorized-numpy-vs-frameworks-cpu.png?raw=true "results/charts-en/vectorized-numpy-vs-frameworks-cpu.png")

### Numpy vs ML Frameworks (GPU and CPU)

Same as above, but compare on GPU as well

![results/charts-en/vectorized-numpy-vs-frameworks-gpu.png](results/charts-en/vectorized-numpy-vs-frameworks-gpu.png?raw=true "results/charts-en/vectorized-numpy-vs-frameworks-gpu.png")

### ML Framework GPU vs Loop CPU

Comparing frameworks running on GPU with naive C++ loop running on CPU.

![results/charts-en/frameworks-gpu-vs-c++-cpu.png](results/charts-en/frameworks-gpu-vs-c++-cpu.png?raw=true "results/charts-en/frameworks-gpu-vs-c++-cpu.png")

### C++ Parallel APIs (CPU)

Comparing naive C++ loop with OpenCL and OpenMP on CPU.

![results/charts-en/parallel-c++-cpu.png](results/charts-en/parallel-c++-cpu.png?raw=true "results/charts-en/parallel-c++-cpu.png")

### C++ CPU vs GPU

Comparing naive C++ loop with CUDA and OpenCL on GPU

![results/charts-en/c++-cpu-vs-gpu.png](results/charts-en/c++-cpu-vs-gpu.png?raw=true "results/charts-en/c++-cpu-vs-gpu.png")

### OpenCL vs PyOpenCL (CPU & GPU)

Comparing C++ OpenCL with OpenCL Python wrapper.

![results/charts-en/pyopencl-vs-opencl.png](results/charts-en/pyopencl-vs-opencl.png?raw=true "results/charts-en/pyopencl-vs-opencl.png")

### PyCUDA vs C++ (GPU)

Comparing PyCUDA (Python CUDA wrapper) with C++ OpenCL on GPU and C++ naive loop

![results/charts-en/pycuda-vs-c++.png](results/charts-en/pycuda-vs-c++.png?raw=true "results/charts-en/pycuda-vs-c++.png")

### Tensorflow: Python vs C++ (GPU)

Comparing Tensorflow C++ and Python performance

![results/charts-en/tensorflow-python-vs-c++.png](results/charts-en/tensorflow-python-vs-c++.png?raw=true "results/charts-en/tensorflow-python-vs-c++.png")

### Conclusion

All results, excluding Python and R loops

![results/charts-en/conclusion.png](results/charts-en/conclusion.png?raw=true "results/charts-en/conclusion.png")

