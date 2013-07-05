#include <iostream>
#include <fstream>

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <boost/timer.hpp>
#include <boost/program_options.hpp>

#include "kronos_config.h"

#ifdef MKL_FOUND
    #include <mkl.h>
    #include "mkl_cblas.h"
#endif

#include "matrix_sizes.h"
#include "matrix_mult.h"
#include "mmcublas.h"

// Allocates a matrix with random real_t entries
void random_init(real_t* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = 10*i;
      //  data[i] = rand() / (real_t)RAND_MAX;
}

void summary( std::string name, double s )
{
    double mmflops = 2.0 * (double)WA * (double)HA * (double)WB;

    double gflops = (mmflops * 1.0e-9f) / s;

    printf("[%s] took %6.3f s -> %6.3f GFlops\n",name.c_str(),s,gflops);
}


int main(int argc, char * argv[])
{
  // Declare the supported options.
  boost::program_options::options_description desc("allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("file", boost::program_options::value<std::string>() , "name of the file to create" )
      ("gpu", "run with cuda code")
      ("mkl", "run with mkl blas dgemm")
      ("cpu", "run with native code")
      ("cublas", "run with cuda blas dgemm");

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
  }

  // create matrices -----------------------------------------------------

  /* set seed for rand()*/
  srand(2006);

  unsigned int size_A = WA * HA;
  unsigned int size_B = WB * HB;
  unsigned int size_C = WC * HC;

  unsigned int mem_size_A = sizeof(real_t) * size_A;
  unsigned int mem_size_B = sizeof(real_t) * size_B;
  unsigned int mem_size_C = sizeof(real_t) * size_C;

  real_t* A = (real_t*) malloc(mem_size_A);
  real_t* B = (real_t*) malloc(mem_size_B);
  real_t* C = (real_t*) malloc(mem_size_C);

  /* 2. initialize host memory*/
  random_init(A, size_A);
  random_init(B, size_B);

  // run with native code ---------------------------------------------------

  if (vm.count("cpu"))
  {
    boost::timer ntimer;

    for(unsigned int i=0;i< HA;i++)
    {
      for(unsigned int j=0;j< WB;j++)
      {
        C[i * WC + j] = 0.0;
        for(unsigned int k=0;k< WA;k++)
        {
          C[i * WC + j] +=  A[i * WA + k] * B[k * WB +j];
        }
      }
    }

    summary( "native", ntimer.elapsed() );
  }

  // run with CUDA code -----------------------------------------------------

  if (vm.count("gpu"))
  {
    boost::timer ntimer;

    gpu_mat_mul(A, B, C);

    summary( "gpu", ntimer.elapsed() );
  }

  // run with MKL BLAS code -----------------------------------------------------

#ifdef MKL_FOUND
  if (vm.count("mkl"))
  {
    real_t alpha = 1.0;
    real_t beta = 1.0;

    const int m = WA;
    const int k = HA;
    const int n = HB;

//    std::cout << "max threads : " << MKL_Get_Max_Threads() << std::endl;
//    MKL_Set_Num_Threads(8);
//    mkl_domain_set_num_threads ( 4, MKL_BLAS );

    boost::timer ntimer;

    cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k, alpha, A, k, B, n, beta, C, n);

    summary( "mkl", ntimer.elapsed() );

  }
#endif

  // run with Cuda BLAS code -----------------------------------------------------

  if (vm.count("cublas"))
  {

    mmcublas_init();

    boost::timer ntimer;

    mmcublas_dgemm(A, B, C);

    summary( "cublas", ntimer.elapsed() );

  }

  // write result  -------------------------------------------------------

  if (vm.count("file"))
  {
    std::string filename = vm["file"].as<std::string>();
    std::cout << "writing to " << filename << std::endl;
    std::ofstream fout ( filename.c_str() );
    for( unsigned  int i = 0; i < size_C; i++)
    {
      fout << C[i] << " ";
      if(((i + 1) % WC) == 0 ) fout << "\n";
    }
    fout.close();
  }

  // clean up memory -------------------------------------------------------

  free(A);
  free(B);
  free(C);

  return 0;
}
