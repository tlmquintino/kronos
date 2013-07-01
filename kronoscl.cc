#include <stdio.h>
#include <iostream>
#include <string>
#include <cstdlib>

#include "kronoscl.h"
#include "util.h"

namespace kronos {

//------------------------------------------------------------------------------------------

CL::CL()
{
    printf("Initialize OpenCL object and context\n");
    //setup devices and context
    
    //this function is defined in util.cpp
    //it comes from the NVIDIA SDK example code
    err = oclGetPlatformID(&platform);
    //oclErrorString is also defined in util.cpp and comes from the NVIDIA SDK
    printf("oclGetPlatformID: %s\n", oclErrorString(err));

    // Get the number of GPU devices available to the platform
    // we should probably expose the device type to the user
    // the other common option is CL_DEVICE_TYPE_CPU
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    printf("clGetDeviceIDs (get number of devices): %s\n", oclErrorString(err));


    // Create the device list
    devices = new cl_device_id [numDevices];
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    printf("clGetDeviceIDs (create device list): %s\n", oclErrorString(err));
 
    //create the context
    context = clCreateContext(0, 1, devices, NULL, NULL, &err);

    //for right now we just use the first available device
    //later you may have criteria (such as support for different extensions)
    //that you want to use to select the device
    deviceUsed = 0;
    
    //create the command queue we will use to execute OpenCL commands
    command_queue = clCreateCommandQueue(context, devices[deviceUsed], 0, &err);

    cl_a = 0;
    cl_b = 0;
    cl_c = 0;
}

//------------------------------------------------------------------------------------------

CL::~CL()
{
    printf("Releasing OpenCL memory\n");
    if(kernel)clReleaseKernel(kernel); 
    if(program)clReleaseProgram(program);
    if(command_queue)clReleaseCommandQueue(command_queue);
   
    //need to release any other OpenCL memory objects here
    if(cl_a)clReleaseMemObject(cl_a);
    if(cl_b)clReleaseMemObject(cl_b);
    if(cl_c)clReleaseMemObject(cl_c);

    if(context)clReleaseContext(context);
    
    if(devices)delete(devices);
    printf("OpenCL memory released\n");
}

//------------------------------------------------------------------------------------------

void CL::load(const char* relative_path)
{
 // Program Setup
    int pl;
    size_t program_length;
    printf("load the program\n");
    
    //CL_SOURCE_DIR is set in the CMakeLists.txt
    std::string path(KRONOS_SRC_DIR);
    path += "/" + std::string(relative_path);
    printf("path: %s\n", path.c_str());

    //file_contents is defined in util.cpp
    //it loads the contents of the file at the given path
    char* cSourceCL = file_contents(path.c_str(), &pl);
    //printf("file: %s\n", cSourceCL);
    program_length = (size_t)pl;

    // create the program
    program = clCreateProgramWithSource(context, 1,
                      (const char **) &cSourceCL, &program_length, &err);
    printf("clCreateProgramWithSource: %s\n", oclErrorString(err));

    build();

    //Free buffer returned by file_contents
    free(cSourceCL);
}

//------------------------------------------------------------------------------------------

void CL::build()
{
    // Build the program executable
    
    printf("building the program\n");
    // build the program
    //err = clBuildProgram(program, 0, NULL, "-cl-nv-verbose", NULL, NULL);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    printf("clBuildProgram: %s\n", oclErrorString(err));
	//if(err != CL_SUCCESS){
		cl_build_status build_status;
		err = clGetProgramBuildInfo(program, devices[deviceUsed], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);

		char *build_log;
		size_t ret_val_size;
		err = clGetProgramBuildInfo(program, devices[deviceUsed], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);

		build_log = new char[ret_val_size+1];
		err = clGetProgramBuildInfo(program, devices[deviceUsed], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
		build_log[ret_val_size] = '\0';
		printf("BUILD LOG: \n %s", build_log);
	//}
    printf("program built\n");
}

//------------------------------------------------------------------------------------------

void CL::init()
{
    std::cout << "in " << __FUNCTION__ << std::endl;

    // initialize kernel from the program
    kernel = clCreateKernel( program, "vecmult", &err );
    printf("clCreateKernel: %s\n", oclErrorString(err));

    // initialize our CPU memory arrays, send them to the device and set the kernel arguements
    num = 10;
    data_t *a = new data_t[num];
    data_t *b = new data_t[num];
    for(int i=0; i < num; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    printf("Creating OpenCL arrays\n");
    //our input arrays
    //create our OpenCL buffer for a, copying the data from CPU to the GPU at the same time
    cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(data_t) * num, a, &err);
    //cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(data_t) * num, b, &err);
    //we could do b similar, but you may want to create your buffer and fill it at a different time
    cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(data_t) * num, NULL, &err);
    //our output array
    cl_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(data_t) * num, NULL, &err);

    printf("Pushing data to the GPU\n");
    //push our CPU arrays to the GPU
//    err = clEnqueueWriteBuffer(command_queue, cl_a, CL_TRUE, 0, sizeof(data_t) * num, a, 0, NULL, &event);
//    clReleaseEvent(event); //we need to release events in order to be completely clean (has to do with openclprof)
//
    //push b's data to the GPU
    err = clEnqueueWriteBuffer(command_queue, cl_b, CL_TRUE, 0, sizeof(data_t) * num, b, 0, NULL, &event);
    clReleaseEvent(event);


    // set the arguements of our kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &cl_a);
    err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &cl_b);
    err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &cl_c);

    // wait for the command queue to finish these commands before proceeding
    clFinish(command_queue);

    // clean up allocated space.
    delete[] a;
    delete[] b;

    // for now we make the workgroup size the same as the number of elements in our arrays
    workGroupSize[0] = num;
}

void CL::run()
{
    printf("in runKernel\n");
    //execute the kernel
    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, workGroupSize, NULL, 0, NULL, &event);
    clReleaseEvent(event);
    printf("clEnqueueNDRangeKernel: %s\n", oclErrorString(err));
    clFinish(command_queue);

    // lets check our calculations by reading from the device memory and printing out the results
    data_t c_done[num];
    err = clEnqueueReadBuffer(command_queue, cl_c, CL_TRUE, 0, sizeof(data_t) * num, &c_done, 0, NULL, &event);
    printf("clEnqueueReadBuffer: %s\n", oclErrorString(err));
    clReleaseEvent(event);

    for(int i=0; i < num; i++)
    {
        printf("c_done[%d] = %g\n", i, c_done[i]);
    }
}

} // namespace kronos

