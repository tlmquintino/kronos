#include <fstream>
#include <sstream>

#include <boost/timer.hpp>

#include "kronos_config.h"

#include "Gemm.h"
#include "Endian.h"
#include "MData.h"

using namespace std;

//------------------------------------------------------------------------------------------

kronos::Gemm::Gemm() :
    flops_(0)
{
}

void kronos::Gemm::setup(const boost::filesystem::path& p)
{
    test_ = p;

    /// @todo allocate here the test data
    ///

    mm_ = new MData( 100, 11, 68 );


    // A

//    std::ifstream fa;
//    fa.open( "data/antisymmetric_matrix_01258_latitude_x_truncation_00100_00011", ios::in | ios::binary );
//    MData::load_be_cm( mm_->A , fa );
//    fa.close();

//    // B

//    std::ifstream fb;
//    fb.open( "data/antisymmetric_matrix_01258_truncation_x_field_00011_00068", ios::in | ios::binary );
//    MData::load_be_cm( mm_->B , fb );
//    fb.close();

//    // Cr

    std::ifstream fc;
    fc.open( "data/antisymmetric_matrix_01258_latitude_x_field_00100_00068", ios::in | ios::binary );
    MData::load_be_cm( mm_->Cr , fc );
    fc.close();


//    std::cout << "--- A --------------------------------------------" << std::endl;
//    MData::print( mm_->A , std::cout, 7 );
//    std::cout << "--------------------------------------------------" << std::endl;
//    std::cout << "--- B --------------------------------------------" << std::endl;
//    MData::print( mm_->B , std::cout );
//    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "--- C --------------------------------------------" << std::endl;
    MData::print( mm_->Cr , std::cout, 8 );
    std::cout << "--------------------------------------------------" << std::endl;

    std::cout << "A size:" << mm_->A.size1() * mm_->A.size2() * sizeof(real_t) << std::endl;
    std::cout << "B size:" << mm_->B.size1() * mm_->B.size2() * sizeof(real_t) << std::endl;
    std::cout << "C size:" << mm_->C.size1() * mm_->C.size2() * sizeof(real_t) << std::endl;

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

//------------------------------------------------------------------------------------------
