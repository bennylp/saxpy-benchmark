#include <iostream>
#include <cmath>
#define __CL_ENABLE_EXCEPTIONS
//#define CL_VERSION_1_2
//#include <CL/cl.hpp>		// <-- On Error, use the "extra/cl.hpp" instead
#include "extra/cl.hpp"

const int N = 1 << 26;
const unsigned XVAL = 2.0f;
const unsigned YVAL = 1.0f;
const unsigned AVAL = 3.0f;

#include <chrono>
class saxpy_timer {
public:
  saxpy_timer() {
    reset();
  }
  void reset() {
    t0_ = std::chrono::high_resolution_clock::now();
  }
  double elapsed(bool reset_timer = false) {
    std::chrono::high_resolution_clock::time_point t =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<
        std::chrono::duration<double>>(t - t0_);
    if (reset_timer)
      reset();
    return time_span.count();
  }
  double elapsed_msec(bool reset_timer = false) {
    return elapsed(reset_timer) * 1000;
  }
private:
  std::chrono::high_resolution_clock::time_point t0_;
};

static void saxpy_verify(const float *y)
{
  float err = 0.0;
  for (size_t i = 0; i < N; ++i)
    err = err + fabs(y[i] - (AVAL * XVAL + YVAL));
  std::cout << "Errors: " << err << std::endl;
}

int main(int argc, const char *argv[]) {
  try {
    float *cpuX = new float[N], *cpuY = new float[N];
    for (size_t i = 0; i < N; ++i) {
      cpuX[i] = XVAL; cpuY[i] = YVAL;
    }

    // Time the CPU version
    saxpy_timer timer;
    for (size_t i=0; i < N; ++i)
      cpuY[i] += AVAL * cpuX[i];
    std::cout << "CPU Elapsed: " << timer.elapsed_msec() << " ms\n";
    saxpy_verify(cpuY);

    // Reset Y
    for (size_t i=0; i<N; ++i)
      cpuY[i] = YVAL;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
      throw cl::Error(-1, "No platforms found");

    std::vector<cl::Device> devices;
    platforms[1].getDevices(CL_DEVICE_TYPE_CPU, &devices);
    if (devices.empty())
      throw cl::Error(-1, "Device not found");

    cl::Device default_device = devices[0];
    std::cout << "Using " << default_device.getInfo<CL_DEVICE_VENDOR>()
              << " " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    cl::Context context( { default_device });

    std::string kernel_code =
        "__kernel void saxpy(const float alpha,"
        "                    __global const float* X,"
        "                    __global float* Y) {"
        "    int i = get_global_id(0); "
        "    Y[i] += alpha * X[i];"
        "}";
    cl::Program::Sources sources;
    sources.push_back( { kernel_code.c_str(), kernel_code.length() });

    cl::Program program(context, sources);
    program.build( { default_device });

    cl::Buffer devX(context, CL_MEM_READ_WRITE, N * sizeof(float));
    cl::Buffer devY(context, CL_MEM_READ_WRITE, N * sizeof(float));

    cl::CommandQueue queue(context, default_device);
    queue.enqueueWriteBuffer(devX, CL_TRUE, 0, N * sizeof(float), cpuX);
    queue.enqueueWriteBuffer(devY, CL_TRUE, 0, N * sizeof(float), cpuY);

    cl::Kernel kernel(program, "saxpy");
    kernel.setArg(0, (float) AVAL);
    kernel.setArg(1, devX);
    kernel.setArg(2, devY);

    timer.reset();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N));
    queue.finish();

    std::cout << "OCL Elapsed: " << timer.elapsed_msec() << " ms\n";

    queue.enqueueReadBuffer(devY, CL_TRUE, 0, N * sizeof(float), cpuY);
    saxpy_verify(cpuY);

    delete[] cpuX;
    delete[] cpuY;

  } catch (cl::Error err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")\n";
  }
}
