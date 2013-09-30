#include <vector>
#include <algorithm> 
#include <iostream>

// #include "boost/date_time/posix_time/posix_time.hpp"

#include <Eigen/Dense>

#include "utils.h"

using namespace Eigen;

int main()
{
    std::ostringstream osi;
    std::ostringstream osf;

    const size_t n = 10;
    const size_t k = 8;
    const size_t m = 6;

    MatrixXd A(n,k);
    MatrixXd B(k,m);
    MatrixXd C(n,m);

    A.setRandom();
    B.setRandom();

//    boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::universal_time();

    C = A * B;

    osi << C << std::endl;

    // verification

    MatrixXd Cr(n,m);
    Cr.setZero();

    for(size_t i=0; i<n; ++i)
        for(size_t j=0; j<m; ++j)
            for(size_t l=0; l<k; ++l)
                Cr(i,j) +=  A(i,l) * B(l,j);

    osf << Cr << std::endl;

//    boost::posix_time::time_duration dt = boost::posix_time::microsec_clock::universal_time() - t1;
//    std::cout << "eigen dgemm: " << ( flops(n,k,m) * 1E-9 ) / ( dt.total_microseconds() / 1E6 ) << "gflops" << std::endl;

    if( osi.str() == osf.str() )
    {
        std::cout << "OK" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "FAIL" << std::endl;
        return -1;
    }

    return 0;
}
