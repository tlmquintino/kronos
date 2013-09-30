#ifndef kronos_eigen_utils
#define kronos_eigen_utils

#include <string>
#include <sys/types.h>

#include <Eigen/Dense>

//-------------------------------------------------------------------------------------------

std::string sep = "\n----------------------------------------\n";

Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

//-------------------------------------------------------------------------------------------

double flops( size_t n, size_t k, size_t m) { return 2.0 * (double)m * (double)k * (double)n; }

//-------------------------------------------------------------------------------------------

double rand01() // returns double between 0.0 and 1.0
{
    return (double)rand() / (double)RAND_MAX;
}

//-------------------------------------------------------------------------------------------

template < typename MatrixType >
void init_matrix( MatrixType& m )
{
    for( size_t i = 0; i < m.rows(); ++i )
        for( size_t j = 0; j < m.cols(); ++j )
            m(i,j) = rand01();
}

//-------------------------------------------------------------------------------------------

#endif // kronos_eigen_utils
