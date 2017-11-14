#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "saxpy.h"

// For some docs:
// - http://github.khronos.org/OpenCL-CLHPP/
// - http://simpleopencl.blogspot.co.id/2013/06/tutorial-simple-start-with-opencl-and-c.html

#define __CL_ENABLE_EXCEPTIONS
//#define CL_VERSION_1_2
//#include "extra/cl.hpp"
#include <CL/cl.hpp>

static std::string dev_type_name(unsigned dev_type)
{
   std::string ret;
   if (dev_type & CL_DEVICE_TYPE_CPU) ret += "cpu ";
   if (dev_type & CL_DEVICE_TYPE_GPU) ret += "gpu ";
   if (dev_type & CL_DEVICE_TYPE_ACCELERATOR) ret += "accel ";
   if (dev_type & CL_DEVICE_TYPE_CUSTOM) ret += "custom ";
   return ret;
}

// Eumerate and select device
// where: "cpu", "gpu", or ""
static cl::Device select_device(const std::string &where)
{
   std::vector<cl::Platform> platforms;
   cl::Platform::get(&platforms);
   if (platforms.empty())
      throw cl::Error(-1, "No platforms found");

   cl::Device selected_dev;

   for (auto &plat : platforms) {
      std::cout << "Platform \"" << plat.getInfo<CL_PLATFORM_NAME>() << "\". Devices:\n";

      std::vector<cl::Device> devices;
      plat.getDevices(CL_DEVICE_TYPE_ALL, &devices);
      if (devices.size() == 0) {
         std::cout << "  No devices found.\n";
         continue;
      }

      for (auto &dev : devices) {
         auto dev_type = dev.getInfo<CL_DEVICE_TYPE>();
         std::cout << " - [" << dev_type_name(dev_type) << "] "
                   << dev.getInfo<CL_DEVICE_VENDOR>() << ": "
                   << dev.getInfo<CL_DEVICE_NAME>() << std::endl;
         std::cout << "   (Max compute units: "
                   << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
                   << ", max work group size: "
                   << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << ")\n";

         if (selected_dev())
            continue;

         if (where == "cpu" && dev_type == CL_DEVICE_TYPE_CPU) {
            selected_dev = dev;
         } else if (where == "gpu" && dev_type == CL_DEVICE_TYPE_GPU) {
            selected_dev = dev;
         }
      }
      std::cout << std::endl;
   }

   if (where.empty())
      selected_dev = cl::Device::getDefault();

   if (!selected_dev())
      throw cl::Error(-1, "Device not found");

   return selected_dev;

}

int main(int argc, const char *argv[])
{
   try {
      cl::Device default_device = select_device(argc > 1 ? argv[1] : "");
      std::cout << "Using " << default_device.getInfo<CL_DEVICE_VENDOR>() << " "
                << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;

      cl::Context context({default_device});
      cl::Program::Sources sources;
      std::string kernel_code =
            "__kernel void saxpy(float alpha,"
            "                    __global const float* X,"
            "                    __global float* Y)"
            "{"
            "    int i = get_global_id(0); "
            "    Y[i] += alpha * X[i];"
            "}";
      sources.push_back( { kernel_code.c_str(), kernel_code.length() });

      cl::Program program(context, sources);
      program.build( {default_device} );

      cl::Buffer dev_x(context, CL_MEM_READ_ONLY, N * sizeof(float));
      cl::Buffer dev_y(context, CL_MEM_READ_WRITE, N * sizeof(float));

      float *host_x = new float[N], *host_y = new float[N];
      for (size_t i = 0; i < N; ++i) {
         host_x[i] = XVAL;
         host_y[i] = YVAL;
      }

      cl::CommandQueue queue(context, default_device);
      queue.enqueueWriteBuffer(dev_x, CL_TRUE, 0, N * sizeof(float), host_x);
      queue.enqueueWriteBuffer(dev_y, CL_TRUE, 0, N * sizeof(float), host_y);

      cl::Kernel kernel(program, "saxpy");
      saxpy_timer timer;

#if 0	// Nicer but only available on newer OpenCL (cl2.hpp?)
      cl::KernelFunctor saxpy( kernel, queue, cl::NullRange, cl::NDRange(N), cl::NullRange);
      saxpy(AVAL, dev_x, dev_y);
#else
      kernel.setArg(0, (float) AVAL);
      kernel.setArg(1, dev_x);
      kernel.setArg(2, dev_y);

      queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N));
      queue.finish();
#endif
      std::cout << "Elapsed: " << timer.elapsed_msec() << " ms\n";

      queue.enqueueReadBuffer(dev_y, CL_TRUE, 0, N * sizeof(float), host_y);
      saxpy_verify(host_y);

      delete[] host_x;
      delete[] host_y;

   } catch (cl::Error err) {
      std::cerr << "ERROR: " << err.what() << "(code: " << err.err() << ")\n";
   }
}
