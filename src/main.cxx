#include <iostream>
#include <cassert>
#include <limits>
#include <iomanip>
#include <numeric>


#include <boost/shared_ptr.hpp>
#include <boost/timer.hpp>
#include <boost/program_options.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

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
using namespace boost::filesystem;

//-----------------------------------------------------------------------------

struct WaveN
{
    WaveN(){}

    WaveN( size_t w,
           size_t l = std::numeric_limits<size_t>::max(),
           size_t t = std::numeric_limits<size_t>::max()) : wn(w), lat(l), trc(t)
    {}

    size_t wn;
    size_t lat;
    size_t trc;
    std::vector<size_t> fields;
};

//-----------------------------------------------------------------------------

void run( boost::shared_ptr<Gemm> gemm, const boost::filesystem::path& p )
{
    // some issues with these set of fields ???

    // wn [6,10,11] lat 639 trc 635 f[408]
    // wn 7 lat 640 trc 637 f[68,408]

    if( !exists(p) ) printf("path %s does not exist\n", p.string().c_str() ), ::exit(1);

    if( !is_directory(p) ) printf("path %s is not a directory\n", p.string().c_str() ), ::exit(1);

    //  e.g.  antisymmetric_matrix_00006_latitude_x_truncation_00640_00637
    const boost::regex latfilter( "antisymmetric_matrix_(\\d+)_latitude_x_truncation_(\\d+)_(\\d+)" );

    //  e.g.  antisymmetric_matrix_00000_truncation_x_field_00640_00172
    const boost::regex trcfilter( "antisymmetric_matrix_(\\d+)_truncation_x_field_(\\d+)_(\\d+)" );

    std::map< size_t, WaveN > matches;

    for( directory_iterator i( p ); i != directory_iterator(); ++i )
    {
        if( !is_regular_file( i->status() ) ) continue; /* skip non regular files */

        const path f = i->path().filename();

        boost::smatch sm;
        if( boost::regex_match( f.string(), sm, latfilter ) )
        {
            assert( sm.size() == 4 );

            //        std::copy( sm.begin(), sm.end(), std::ostream_iterator<std::string>(std::cout, "\n"));

            size_t wn = boost::lexical_cast<size_t>( sm[1] );

            if( matches.find(wn) == matches.end() )
                matches[wn] = WaveN( wn );
            matches[wn].lat = boost::lexical_cast<size_t>(sm[2]);
            matches[wn].trc = boost::lexical_cast<size_t>(sm[3]);
        }

        if( boost::regex_match( f.string(), sm, trcfilter ) )
        {
            assert( sm.size() == 4 );

//            std::copy( sm.begin(), sm.end(), std::ostream_iterator<std::string>(std::cout, "\n"));

            size_t wn = boost::lexical_cast<size_t>( sm[1] );

            if( matches.find(wn) == matches.end() )
                matches[wn] = WaveN( wn );
            matches[wn].trc = boost::lexical_cast<size_t>(sm[2]);
            matches[wn].fields.push_back( boost::lexical_cast<size_t>(sm[3]) );
        }
    }

    std::map< size_t, WaveN >::iterator it = matches.begin();
    for( ; it != matches.end(); ++it )
    {

        size_t wn  = it->second.wn;
        size_t lat = it->second.lat;
        size_t trc = it->second.trc;

        std::vector<size_t> fields = it->second.fields;

        std::sort( fields.begin(), fields.end() );

//#define FIELD_BY_FIELD

#ifdef  FIELD_BY_FIELD

        for( size_t f = 0; f < fields.size(); ++f )
        {
            std::vector<size_t> f1;
            f1 += fields[f];

            std::cout << "wn "  << std::setw(5) << wn << "   "
                      << " C (" << std::setw(5) << lat << "," << std::setw(5) << fields[f] << ") "
                      << " A (" << std::setw(5) << lat << "," << std::setw(5) << trc << ") "
                      << " B (" << std::setw(5) << trc << "," << std::setw(5) << fields[f] << ") "
                      << std::flush;

            gemm->setup(p,wn,lat,trc,f1);

            gemm->run();

            gemm->verify();

            gemm->teardown();

            std::cout << gemm->summary() << std::endl;
        }
#else
        size_t sumf = std::accumulate(fields.begin(),fields.end(),0);

//        std::cout << "wn "  << std::setw(5) << wn << "   "
//                  << " C (" << std::setw(5) << lat << "," << std::setw(5) << sumf << ") "
//                  << " A (" << std::setw(5) << lat << "," << std::setw(5) << trc << ") "
//                  << " B (" << std::setw(5) << trc << "," << std::setw(5) << sumf << ") "
//                  << std::flush;

        std::cout << "wn, "   << std::setw(5) << wn << ", "
                  << "lat, "  << std::setw(5) << lat << ", "
                  << "trc, "  << std::setw(5) << trc << ", "
                  << "flds, " << std::setw(5) << sumf << ", "
                  << std::flush;

        gemm->setup(p,wn,lat,trc,fields);

        gemm->run();

        gemm->verify();

        gemm->teardown();

        std::cout << gemm->summary() << std::endl;
#endif

    }
}

//-----------------------------------------------------------------------------

int main(int argc, char * argv[])
{
  // Declare the supported options.
  boost::program_options::options_description desc("allowed options");
  desc.add_options()
          ("help", "produce help message")
          ("test",   boost::program_options::value<std::string>() , "directory with test data" )
          ("align",  boost::program_options::value<size_t>() , "align to bytes" )
          ("steps",  boost::program_options::value<size_t>() , "nb steps" )
          ("threads",boost::program_options::value<size_t>() , "nb threads" )
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

  boost::shared_ptr<Gemm> gemm;

  if( vm.count("cpu") )
      gemm.reset(new CpuGemm());

  if( vm.count("cuda") )
#ifdef CUDA_FOUND
    gemm.reset(new CudaGemm());
#else
      std::cerr << "cuda not available -- aborting" << std::endl, ::exit(1);
#endif

  if( vm.count("cublas") )
#ifdef CUDA_FOUND
    gemm.reset(new CublasGemm());
#else
      std::cerr << "cublas not available -- aborting" << std::endl, ::exit(1);
#endif

if( vm.count("viennacl") )
#ifdef ViennaCL_FOUND
   gemm.reset(new ViennaCLGemm());
#else
      std::cerr << "viennacl not available -- aborting" << std::endl, ::exit(1);
#endif

if( vm.count("mkl") )
#ifdef MKL_FOUND
   gemm.reset(new MKLGemm());
#else
      std::cerr << "mkl not available -- aborting" << std::endl, ::exit(1);
#endif

if( vm.count("cl") )
#ifdef OPENCL_FOUND
   gemm.reset(new CLGemm());
#else
      std::cerr << "opencl not available -- aborting" << std::endl, ::exit(1);
#endif

if( vm.count("blas") )
#ifdef BLAS_FOUND
   gemm.reset(new BlasGemm());
#else
      std::cerr << "blas not available -- aborting" << std::endl, ::exit(1);
#endif

    if( !gemm )
    {
        std::cerr << "error: no gemm method chosen" << std::endl;
        std::cout << desc << std::endl;
        ::exit(1);
    }

    if( vm.count("align") )
        gemm->align_to( vm["align"].as<size_t>() );

    if( vm.count("steps") )
        gemm->steps( vm["steps"].as<size_t>() );

    if( vm.count("threads") )
        gemm->threads( vm["threads"].as<size_t>() );

    run( gemm, tpath );
}
