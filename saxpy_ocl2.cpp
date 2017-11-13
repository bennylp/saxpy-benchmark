// https://fastkor.wordpress.com/2012/07/22/example-opencl-boilerplate/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "saxpy.h"


void exitOnFail(cl_int status, const char* message)
{
    if (CL_SUCCESS != status)
    {
        printf("error: %s\n", message);
        exit(-1);
    }
}

// May specify "cpu" or "gpu" to force hardware selection
int main(int argc, char *argv[])
{
    // return code used by OpenCL API
    cl_int status;

    // wait event synchronization handle used by OpenCL API
    cl_event event;

    const char *wanted = argv[1];

    ////////////////////////////////////////
    // OpenCL platforms

    // determine number of platforms
    cl_uint numPlatforms;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    exitOnFail(status, "number of platforms");

    // get platform IDs
#define MAX_PLATFORMS	16
    cl_platform_id platformIDs[MAX_PLATFORMS];
    status = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    exitOnFail(status, "get platform IDs");

    ////////////////////////////////////////
    // OpenCL devices

    // look for a CPU and GPU compute device
    cl_platform_id cpuPlatformID, gpuPlatformID;
    cl_device_id cpuDeviceID, gpuDeviceID;
    int isCPU = 0, isGPU = 0;

    // iterate over platforms
    for (size_t i = 0; i < numPlatforms; i++)
    {
        // determine number of devices for a platform
        cl_uint numDevices;
        status = clGetDeviceIDs(platformIDs[i],
                                CL_DEVICE_TYPE_ALL,
                                0,
                                NULL,
                                &numDevices);
        if (CL_SUCCESS == status)
        {
            // get device IDs for a platform
#define MAX_DEVICES	128
            cl_device_id deviceIDs[MAX_DEVICES];
            status = clGetDeviceIDs(platformIDs[i],
                                    CL_DEVICE_TYPE_ALL,
                                    numDevices,
                                    deviceIDs,
                                    NULL);
            if (CL_SUCCESS == status)
            {
                // iterate over devices
                for (size_t j = 0; j < numDevices; j++)
                {
                    cl_device_type deviceType;
                    status = clGetDeviceInfo(deviceIDs[j],
                                             CL_DEVICE_TYPE,
                                             sizeof(cl_device_type),
                                             &deviceType,
                                             NULL);
                    if (CL_SUCCESS == status)
                    {
                		printf("Found %s %lx\n",
                			((CL_DEVICE_TYPE_GPU & deviceType) ? "gpu" :
                				(CL_DEVICE_TYPE_CPU & deviceType) ? "cpu" : "other"),
			       (long)deviceIDs[j]);

                        // first CPU device
                        if (!isCPU && (CL_DEVICE_TYPE_CPU & deviceType))
                        {
                            isCPU = 1;
                            cpuPlatformID = platformIDs[i];
                            cpuDeviceID = deviceIDs[j];
                        }

                        // first GPU device
                        if (!isGPU && (CL_DEVICE_TYPE_GPU & deviceType))
                        {
                            isGPU = 1;
                            gpuPlatformID = platformIDs[i];
                            gpuDeviceID = deviceIDs[j];
                        }
                    }
                }
            }
        }
    }

    // pick GPU device if it exists, otherwise use CPU
    cl_platform_id platformID;
    cl_device_id deviceID;
    if (isGPU && (!wanted || strcmp(wanted, "gpu")==0))
    {
        platformID = gpuPlatformID;
        deviceID = gpuDeviceID;
        printf("Launching on gpu %lx\n", (long)deviceID);
    }
    else if (isCPU && (!wanted || strcmp(wanted, "cpu")==0))
    {
        platformID = cpuPlatformID;
        deviceID = cpuDeviceID;
        printf("Launching on cpu %lx\n", (long)deviceID);
    }
    else
    {
        // no devices found
        exitOnFail(CL_DEVICE_NOT_FOUND, "no devices found");
    }

    std::cout << "N: " << N << std::endl;


    ////////////////////////////////////////
    // OpenCL context
    saxpy_timer timer_total;

    cl_context_properties props[] = { CL_CONTEXT_PLATFORM,
                                      (cl_context_properties) platformID,
                                      0 };
    cl_context context = clCreateContext(props,
                                         1,
                                         &deviceID,
                                         NULL,
                                         NULL,
                                         &status);
    exitOnFail(status, "create context");

    ////////////////////////////////////////
    // OpenCL command queue

    cl_command_queue queue = clCreateCommandQueue(context,
                                                  deviceID,
                                                  0,
                                                  &status);
    exitOnFail(status, "create command queue");

    ////////////////////////////////////////
    // OpenCL buffers

    // N x 1 row major array buffers
    float *cpuX, *cpuY;

    cpuX = new float[N];
    cpuY = new float[N];

    // initialize array data
    for (size_t i = 0; i < N; i++)
    {
        cpuX[i] = XVAL;
        cpuY[i] = YVAL;
    }

    // second argument: memory buffer object for X
    cl_mem memX = clCreateBuffer(context,
                                 CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                 N * sizeof(float),
                                 cpuX,
                                 &status);
    exitOnFail(status, "create buffer for X");

    // third argument: memory buffer object for Y
    cl_mem memY = clCreateBuffer(context,
                                 CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                 N * sizeof(float),
                                 cpuY,
                                 &status);
    exitOnFail(status, "create buffer for Y");


    ////////////////////////////////////////
    // OpenCL move buffer data to device

    // data transfer for array X
    status = clEnqueueWriteBuffer(queue,
                                  memX,
                                  CL_FALSE,
                                  0,
                                  N * sizeof(float),
                                  cpuX,
                                  0,
                                  NULL,
                                  &event);
    exitOnFail(status, "write X to device");
    status = clWaitForEvents(1, &event);
    exitOnFail(status, "wait for write X to device");
    clReleaseEvent(event);

    // data transfer for array Y
    status = clEnqueueWriteBuffer(queue,
                                  memY,
                                  CL_FALSE,
                                  0,
                                  N * sizeof(float),
                                  cpuY,
                                  0,
                                  NULL,
                                  &event);
    exitOnFail(status, "write Y to device");
    status = clWaitForEvents(1, &event);
    exitOnFail(status, "wait for write Y to device");
    clReleaseEvent(event);


    ////////////////////////////////////////
    // OpenCL program and kernel

    // saxpy: Y = alpha * X + Y
    const char *kernelSrc[] = {
        "__kernel void saxpy(const float alpha,",
        "                    __global const float* X,",
        "                    __global float* Y)",
        "{",
        "    Y[get_global_id(0)] += alpha * X[get_global_id(0)];",
        "}" };

    // a program can have multiple kernels
    cl_program program = clCreateProgramWithSource(
                             context,
                             sizeof(kernelSrc)/sizeof(const char*),
                             kernelSrc,
                             NULL,
                             &status);
    exitOnFail(status, "create program");

    // compile the program
    status = clBuildProgram(program, 1, &deviceID, "", NULL, NULL);
    exitOnFail(status, "build program");

    // one kernel from the program
    cl_kernel kernel = clCreateKernel(program, "saxpy", &status);
    exitOnFail(status, "create kernel");


    ////////////////////////////////////////
    // OpenCL kernel arguments

    // first argument: a scalar float
    float alpha = AVAL;

    // set first argument
    status = clSetKernelArg(kernel, 0, sizeof(float), &alpha);
    exitOnFail(status, "set kernel argument alpha");

    // set second argument
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memX);
    exitOnFail(status, "set kernel argument X");

    // set third argument
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memY);
    exitOnFail(status, "set kernel argument Y");

    ////////////////////////////////////////
    // OpenCL enqueue kernel and wait

    // N work-items in groups of 4
    const size_t groupsize = 4;
    const size_t global[] = { N }/*, local[] = { groupsize }*/;
    const size_t *local = NULL;


    // Start timer
    saxpy_timer timer;

    // enqueue kernel
    status = clEnqueueNDRangeKernel(queue,
                                    kernel,
                                    sizeof(global)/sizeof(size_t),
                                    NULL,
                                    global,
                                    local,
                                    0,
                                    NULL,
                                    &event);
    exitOnFail(status, "enqueue kernel");

    // wait for kernel, this forces execution
    status = clWaitForEvents(1, &event);

    std::cout << "Elapsed: " << timer.elapsed_msec() << " ms" << std::endl;

    exitOnFail(status, "wait for enqueue kernel");
    clReleaseEvent(event);

    ////////////////////////////////////////
    // OpenCL read back buffer from device

    // data transfer for array Y
    status = clEnqueueReadBuffer(queue,
                                 memY,
                                 CL_FALSE,
                                 0,
                                 N * sizeof(float),
                                 cpuY,
                                 0,
                                 NULL,
                                 &event);
    exitOnFail(status, "read Y from device");
    status = clWaitForEvents(1, &event);
    exitOnFail(status, "wait for read Y from device");
    clReleaseEvent(event);

    std::cout << "Total time: " << timer_total.elapsed_msec() << " ms" << std::endl;

    ////////////////////////////////////////
    // OpenCL cleanup

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(memX);
    clReleaseMemObject(memY);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // print computed result
    saxpy_verify(cpuY);
    delete [] cpuX;
    delete [] cpuY;

    exit(0);
}
