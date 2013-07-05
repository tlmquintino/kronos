#include <iostream>
#include <fstream>

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <boost/timer.hpp>
#include <boost/program_options.hpp>

#include "kronos_config.h"

#include "matrix_sizes.h"
#include "matrix_mult.h"

// Allocates a matrix with random real_t entries
void random_init(real_t* data, int size)
{
    for (int i = 0; i < size; ++i)
      data[i] = 10*i;
    //  data[i] = rand() / (real_t)RAND_MAX;
}


int main(int argc, char * argv[])
{

  // Declare the supported options.
  boost::program_options::options_description desc("allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("file", boost::program_options::value<std::string>() , "name of the file to create" )
      ("gpu", "run with cuda code")
      ("cpu", "run with native code");

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

  real_t* h_A = (real_t*) malloc(mem_size_A);
  real_t* h_B = (real_t*) malloc(mem_size_B);
  real_t* h_C = (real_t*) malloc(mem_size_C);

  /* 2. initialize host memory*/
  random_init(h_A, size_A);
  random_init(h_B, size_B);

  // run with native code ---------------------------------------------------

  if (vm.count("cpu"))
  {

    boost::timer ntimer;

    for(unsigned int i=0;i< HA;i++)
    {
      for(unsigned int j=0;j< WB;j++)
      {
        h_C[i * WC + j] = 0.0;
        for(unsigned int k=0;k< WA;k++)
        {
          h_C[i * WC + j] += h_A[i * WA + k] * h_B[k * WB +j];
        }
      }
    }

    printf("[native] time: %6.3f seconds\n", ntimer.elapsed() );

  }

  // run with CUDA code -----------------------------------------------------

  if (vm.count("gpu"))
  {

    boost::timer Timer;

      gpu_mat_mul(h_A, h_B, h_C);

    printf("[cuda] time: %6.3f seconds\n", Timer.elapsed() );

  }

  // write result  -------------------------------------------------------

  if (vm.count("file"))
  {
    std::string filename = vm["file"].as<std::string>();
    std::cout << "writing to " << filename << std::endl;
    std::ofstream fout ( filename.c_str() );
    for( unsigned  int i = 0; i < size_C; i++)
    {
      fout << h_C[i] << " ";
      if(((i + 1) % WC) == 0 ) fout << "\n";
    }
    fout.close();
  }

  // clean up memory -------------------------------------------------------

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
