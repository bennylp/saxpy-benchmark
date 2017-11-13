// http://github.khronos.org/OpenCL-CLHPP/
// http://simpleopencl.blogspot.co.id/2013/06/tutorial-simple-start-with-opencl-and-c.html
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "saxpy.h"

#define __CL_ENABLE_EXCEPTIONS
//#define CL_VERSION_1_2
//#include "extra/cl.hpp"
#include <CL/cl.hpp>

static std::string dev_type_name(unsigned dev_type)
{
    std::string ret;

    if (dev_type & CL_DEVICE_TYPE_CPU)
	ret += "cpu ";
    if (dev_type & CL_DEVICE_TYPE_GPU)
	ret += "gpu ";
    if (dev_type & CL_DEVICE_TYPE_ACCELERATOR)
    	ret += "accel ";
    if (dev_type & CL_DEVICE_TYPE_CUSTOM)
        ret += "custom ";
    return ret;
}

// Eumerate and select device
// wanted: "cpu" or "gpu"
static cl::Device select_device(const std::string &wanted)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size()==0)
	throw cl::Error(-1, "No platforms found");

    cl::Device selected_dev;
    bool has_device = false;

    for (auto &plat : platforms) {
	std::cout << "Platform \"" << plat.getInfo<CL_PLATFORM_NAME>() << "\". Devices:\n";

	std::vector<cl::Device> devices;
	plat.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if (devices.size()==0) {
	    std::cout << "  No devices found.\n";
	    continue;
	}

	for (auto &dev : devices) {
	    auto dev_type = dev.getInfo<CL_DEVICE_TYPE>();
	    std::cout << " - [" << dev_type_name(dev_type)
		      << "] " << dev.getInfo<CL_DEVICE_VENDOR>()
		      << ": " << dev.getInfo<CL_DEVICE_NAME>()
		      << std::endl;
	    std::cout << "   (Max compute units: " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
		      << ", max work group size: " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
		      << ")" << std::endl;

	    if (has_device)
		continue;

	    if (wanted == "cpu" && dev_type == CL_DEVICE_TYPE_CPU) {
		selected_dev = dev;
		has_device = true;
	    } else if (wanted == "gpu" && dev_type == CL_DEVICE_TYPE_GPU) {
		selected_dev = dev;
		has_device = true;
	    }
	}
	std::cout << std::endl;
    }

    if (wanted.empty()) {
	selected_dev = cl::Device::getDefault();
	has_device = true;
    }

    if (!has_device)
	throw cl::Error(-1, "Device not found");

    return selected_dev;

}

int main(int argc, const char *argv[])
{
    cl_int err = CL_SUCCESS;

    try {
	cl::Device default_device = select_device(argc > 1 ? argv[1] : "");
	std::cout << "Using " << default_device.getInfo<CL_DEVICE_VENDOR>()
		  << " " << default_device.getInfo<CL_DEVICE_NAME>()
		  << std::endl;

	cl::Context context({default_device});
	cl::Program::Sources sources;
	std::string kernel_code =
	    "__kernel void saxpy(const float alpha,"
	    "                    __global const float* X,"
	    "                    __global float* Y)"
	    "{"
	    "    int i = get_global_id(0); "
	    "    Y[i] += alpha * X[i];"
	    "}" ;
	sources.push_back({kernel_code.c_str(),kernel_code.length()});

	cl::Program program(context, sources);
	program.build({default_device});

	cl::Buffer devX(context, CL_MEM_READ_WRITE, N * sizeof(float));
	cl::Buffer devY(context, CL_MEM_READ_WRITE, N * sizeof(float));

	float *cpuX = new float[N];
	float *cpuY = new float[N];

	for (size_t i=0; i<N; ++i) {
	    cpuX[i] = XVAL;
	    cpuY[i] = YVAL;
	}

	cl::CommandQueue queue(context, default_device);
	queue.enqueueWriteBuffer(devX, CL_TRUE, 0, N*sizeof(float), cpuX);
	queue.enqueueWriteBuffer(devY, CL_TRUE, 0, N*sizeof(float), cpuY);

	cl::Kernel kernel(program, "saxpy");

	saxpy_timer timer;

#if 0	// Nice but only available on newer OpenCL (cl2.hpp?)
	cl::KernelFunctor saxpy( kernel, queue, cl::NullRange, cl::NDRange(N), cl::NullRange);
	saxpy(AVAL, devX, devY);
#else
	kernel.setArg(0, (float)AVAL);
	kernel.setArg(1, devX);
	kernel.setArg(2, devY);


	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N));
	queue.finish();
#endif
	std::cout << "Elapsed: " << timer.elapsed_msec() << " ms" << std::endl;

	queue.enqueueReadBuffer( devY, CL_TRUE, 0, N*sizeof(float), cpuY);
	saxpy_verify(cpuY);

	delete [] cpuX;
	delete [] cpuY;

    } catch (cl::Error err) {
	std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
    }
}
