#include <iostream>

#include <boost/shared_ptr.hpp>
#include <boost/timer.hpp>
#include <boost/program_options.hpp>
#include <boost/assign/std/vector.hpp>

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

#ifdef BLAS_FOUND
#include "BlasGemm.h"
#endif

using namespace kronos;
using namespace boost::assign;

//-----------------------------------------------------------------------------

void run( Gemm* gemm, const boost::filesystem::path& tpath )
{
    std::vector<size_t> fields;
    fields += 2,4,68,70,136,140,340,344,350,408,412,420;

    size_t wn  = 10;
    size_t lat = 639;
    size_t trc = 635;

    gemm->setup(tpath,wn,lat,trc,fields);

    gemm->run();

    gemm->verify();

    gemm->teardown();

    std::cout << gemm->summary() << std::endl;

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
//
          ("cpu",      "run with native code")
          ("cuda",     "run with cuda code")
          ("cublas",   "run with cuda blas dgemm")
          ("mkl",      "run with mkl blas dgemm")
          ("viennacl", "run with viennacl dgemm")
          ("cl",       "run with opencl code")
          ("blas",     "run with blas code")
          ;

  boost::program_options::variables_map vm;

  try
  {
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  }
  catch (...)
  {
      std::cout << desc << std::endl;
      ::exit(1);
  }

  boost::program_options::notify(vm);

  if (vm.count("help")) {
      std::cout << desc << std::endl;
      ::exit(1);
  }

  if(!vm.count("test")) {
      std::cout << "error: no test data directory\n" << std::endl;
      std::cout << desc << std::endl;
      ::exit(1);
  }

  boost::filesystem::path tpath( vm["test"].as<std::string>() );

  if( vm.count("cpu") ) run( new CpuGemm(), tpath );

  if( vm.count("cuda") )
#ifdef CUDA_FOUND
    run( new CudaGemm(), tpath );
#else
      std::cerr << "cuda not available -- aborting" << std::endl, ::exit(1);
#endif

  if( vm.count("cublas") )
#ifdef CUDA_FOUND
    run( new CublasGemm(), tpath );
#else
      std::cerr << "cublas not available -- aborting" << std::endl, ::exit(1);
#endif

if( vm.count("viennacl") )
#ifdef ViennaCL_FOUND
   run( new ViennaCLGemm(), tpath );
#else
      std::cerr << "viennacl not available -- aborting" << std::endl, ::exit(1);
#endif

if( vm.count("mkl") )
#ifdef MKL_FOUND
   run( new MKLGemm(), tpath );
#else
      std::cerr << "mkl not available -- aborting" << std::endl, ::exit(1);
#endif

if( vm.count("cl") )
#ifdef OPENCL_FOUND
   run( new CLGemm(), tpath );
#else
      std::cerr << "opencl not available -- aborting" << std::endl, ::exit(1);
#endif

if( vm.count("blas") )
#ifdef BLAS_FOUND
   run( new BlasGemm(), tpath );
#else
      std::cerr << "blas not available -- aborting" << std::endl, ::exit(1);
#endif

}
