#include <iostream>

#include <boost/shared_ptr.hpp>
#include <boost/timer.hpp>
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

#ifdef ViennaCL_FOUND
#include "ViennaCLGemm.h"
#endif

using namespace kronos;

//-----------------------------------------------------------------------------

void run( Gemm* gemm, const boost::filesystem::path& tpath )
{
    boost::timer total;

    gemm->setup(tpath);

    gemm->run();

    gemm->verify();

    gemm->teardown();

    std::cout << "[" << gemm->name() << "]" << std::endl;
    std::cout << gemm->summary() << std::endl;

    std::cout << "Total time: " << total.elapsed() << " s" << std::endl;

    delete gemm;
}

//-----------------------------------------------------------------------------

int main(int argc, char * argv[])
{
  // Declare the supported options.
  boost::program_options::options_description desc("allowed options");
  desc.add_options()
          ("help", "produce help message")
          ("test",   boost::program_options::value<std::string>() , "directory with test data" )
          ("cpu",      "run with native code")
          ("cuda",     "run with cuda code")
          ("cublas",   "run with cuda blas dgemm")
          ("mkl",      "run with mkl blas dgemm")
          ("viennacl", "run with viennacl dgemm")
          ("cl",       "run with opencl code");

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 1;
  }

  if(!vm.count("test")) {
      std::cout << "error: no test data directory" << std::endl;
      std::cout << desc << std::endl;
  }

  boost::filesystem::path tpath( vm["test"].as<std::string>() );

  if( vm.count("cpu") ) run( new CpuGemm(), tpath );

#ifdef CUDA_FOUND
  if( vm.count("cuda") ) run( new CudaGemm(), tpath );
#endif

#ifdef CUDA_FOUND
  if( vm.count("cublas") ) run( new CublasGemm(), tpath );
#endif

#ifdef ViennaCL_FOUND
  if( vm.count("viennacl") ) run( new ViennaCLGemm(), tpath );
#endif

#ifdef MKL_FOUND
  if( vm.count("mkl") ) run( new MKLGemm(), tpath );
#endif

#ifdef OPENCL_FOUND
  if( vm.count("cl") ) run( new CLGemm(), tpath );
#endif

}
