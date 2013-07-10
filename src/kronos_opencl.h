#ifndef kronos_opencl_h
#define kronos_opencl_h

#include "kronos_config.h"

#ifdef OPENCL_FOUND

#if defined __APPLE__ || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#if defined(cl_khr_fp64)     /* Khronos extension available? */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)   /* AMD extension available? */
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
//#else
//#error no double extension available
#endif
#endif /* OPENCL_FOUND */

struct CLEnv
{
     cl_context        context;
     cl_command_queue  command_queue;
     cl_program        program;
     cl_kernel         kernel;
     cl_device_id*     devices;
     size_t            device_size;
     cl_int            errcode;
     cl_platform_id    cpPlatform;
     cl_device_id      cdDevice;
};

char* opencl_errstring(cl_int err);

char * load_program_source(const char *filename);

void opencl_check_error( cl_int& error_code, const char * file, int line);

#define CALL_CL(e) opencl_check_error( e, #e, __FILE__, __LINE__ )

#endif
