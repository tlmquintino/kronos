#include <iostream>

#include <boost/scoped_ptr.hpp>
#include <boost/program_options.hpp>

#include "kronos_config.h"

//-----------------------------------------------------------------------------

#include "CpuGemm.h"

#ifdef CUDA_FOUND
#include "CudaGemm.h"
#include "CublasGemm.h"
#endif

#ifdef OPENCL_FOUND
#include "CLGemm.h"
#endif

#ifdef MKL_FOUND
#include "MKLGemm.h"
#endif

using namespace kronos;

//-----------------------------------------------------------------------------

int main(int argc, char * argv[])
{
  // Declare the supported options.
  boost::program_options::options_description desc("allowed options");
  desc.add_options()
          ("help", "produce help message")
          ("out", boost::program_options::value<std::string>() , "name of the file to create" )
          ("out", boost::program_options::value<std::string>() , "name of the file to create" )
          ("cpu", "run with native code")
          ("cuda", "run with cuda code")
          ("mkl", "run with mkl blas dgemm")
          ("cublas", "run with cuda blas dgemm")
          ("cl", "run with opencl code");

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
  }

  boost::scoped_ptr<Gemm> gemm;

  if( vm.count("cpu") )
      gemm.reset( new CpuGemm() );

#ifdef CUDA_FOUND
  if( vm.count("cuda") )
      gemm.reset( new CudaGemm() );
#endif

#ifdef MKL_FOUND
  if( vm.count("mkl") )
      gemm.reset( new MKLGemm() );
#endif

#ifdef CUDA_FOUND
  if( vm.count("cublas") )
      gemm.reset( new CublasGemm() );
#endif

#ifdef OPENCL_FOUND
  if( vm.count("cl") )
      gemm.reset( new CLGemm() );
#endif

  return 0;
}
