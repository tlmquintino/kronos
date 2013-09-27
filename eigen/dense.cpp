#include <vector>
#include <algorithm> 
#include <iostream>

#include "boost/date_time/posix_time/posix_time.hpp"

#include <Eigen/Dense>

double flops( size_t n, size_t k, size_t m) { return 2.0 * (double)m * (double)k * (double)n; }

double rand01() // returns double between 0.0 and 1.0
{
    return (double)rand() / (double)RAND_MAX;
}

template < typename MatrixType >
void init_matrix( MatrixType& m )
{
    for( size_t i = 0; i < m.rows(); ++i )
        for( size_t j = 0; j < m.cols(); ++j )
            m(i,j) = rand01();
}

int main()
{
    const size_t n = 4096;
    const size_t k = 4096;
    const size_t m = 6000;

    std::cout << "allocating ..." << std::endl;

    Eigen::MatrixXd A( n, k);
    Eigen::MatrixXd B( k, m);
    Eigen::MatrixXd C( n, m);

    std::cout << "initing ..." << std::endl;

    init_matrix(A);
    init_matrix(B);

    std::cout << "computing ..." << std::endl;

    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::universal_time();

    C = A * B;

    boost::posix_time::time_duration dt = boost::posix_time::microsec_clock::universal_time() - t1;

    std::cout << "eigen dgemm: " << ( flops(n,k,m) * 1E-9 ) / ( dt.total_microseconds() / 1E6 ) << "gflops" << std::endl;

    return 0;
}
