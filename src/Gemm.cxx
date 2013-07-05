#include <sstream>

#include <boost/timer.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include "kronos_config.h"

#include "Gemm.h"

kronos::Gemm::Gemm() :
    flops_(0)
{
}

struct MM
{
    MM( size_t m, size_t k, size_t n ) :
        m_(m), k_(k), n_(n),
        A(m,k),
        B(k,n),
        C(m,n)
    {
    }

    size_t m_;
    size_t k_;
    size_t n_;

    boost::numeric::ublas::matrix<real_t> A;
    boost::numeric::ublas::matrix<real_t> B;
    boost::numeric::ublas::matrix<real_t> C;
};



void kronos::Gemm::setup(const boost::filesystem::path& p)
{
    test_ = p;

    /// @todo allocate here the test data


    boost::timer t;

    this->initiate();

    timers_.initiate = t.elapsed();
}

void kronos::Gemm::run()
{
     boost::timer t;

     this->compute();

     timers_.compute = t.elapsed();
}

bool kronos::Gemm::verify()
{
}

void kronos::Gemm::teardown()
{
    boost::timer t;

    this->terminate();

    timers_.terminate = t.elapsed();

    /// @todo deallocate here the data
}

std::string kronos::Gemm::summary()
{
    std::ostringstream ret;


    return ret.str();
}
