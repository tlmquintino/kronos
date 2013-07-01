#ifndef kronoscl_h
#define kronoscl_h

#if defined __APPLE__ || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

namespace kronos {

class CL {

public: // types

//    typedef double data_t;
    typedef float data_t;

public: // members

    cl_mem cl_a;
    cl_mem cl_b;
    cl_mem cl_c;
    
    int num;    //the size of our arrays

    size_t workGroupSize[1]; //N dimensional array of workgroup size we must pass to the kernel

public: // methods

    /// Constructor initializes OpenCL context and automatically chooses platform and device
    CL();

    /// Destructor releases OpenCL objects and frees device memory
    ~CL();

    /// loads an OpenCL program from a file
    /// the path is relative to the SAM_SRC_DIR set in CMakeLists.txt
    void load(const char* relative_path);

    /// setup the data for the kernel
    void init();

    /// execute the kernel
    void run();

private:

    cl_platform_id platform; ///< handles for creating an opencl context

    //device variables
    cl_device_id* devices;
    cl_uint numDevices;
    unsigned int deviceUsed;

    cl_context context;

    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;


    cl_int   err;       ///< cl return error code
    cl_event event;

    /// build is called by load
    /// to build runtime executable from a program
    void build();
};

} // namespace kronos

#endif
