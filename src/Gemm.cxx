#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <boost/timer.hpp>

#include "kronos_config.h"

#include "Gemm.h"
#include "Endian.h"
#include "MData.h"

using namespace std;

//------------------------------------------------------------------------------------------

kronos::Gemm::Gemm() :
    norm_L2_(0.),
    copy_into_(0.),
    copy_back_(0.),
    flops_(0.),
    steps_hour_(6),
    forecast_days_(10)
{
    timers_.copy_in  = 0.;
    timers_.compute  = 0.;
    timers_.copy_out = 0.;
}

void kronos::Gemm::setup(const boost::filesystem::path& p)
{
#if 0

    const size_t lat = 1024; ///< latitude
    const size_t trc = 1024; ///< truncation
    const size_t fld = 1024; ///< field

    mm_ = new MData( lat, trc, fld );

#endif

#if 0

    const size_t lat = 2; ///< latitude
    const size_t trc = 3; ///< truncation
    const size_t fld = 4; ///< field

    mm_ = new MData( lat, trc, fld );

    MData::matrix_t& A  = mm_->A;
    MData::matrix_t& B  = mm_->B;
    MData::matrix_t& Cr = mm_->Cr;

//
    A(0,0) = 1;
    A(0,1) = 2;
    A(0,2) = 3;

    A(1,0) = 2;
    A(1,1) = 4;
    A(1,2) = 6;
//
    B(0,0) = 1;
    B(1,0) = 1;
    B(2,0) = 1;

    B(0,1) = 3;
    B(1,1) = 3;
    B(2,1) = 3;

    B(0,2) = 5;
    B(1,2) = 5;
    B(2,2) = 5;

    B(0,3) = 7;
    B(1,3) = 7;
    B(2,3) = 7;
//
    Cr(0,0) = 6;
    Cr(1,0) = 12;

    Cr(0,1) = 18;
    Cr(1,1) = 36;

    Cr(0,2) = 30;
    Cr(1,2) = 60;

    Cr(0,3) = 42;
    Cr(1,3) = 84;
//
#endif

#if 1
    const size_t  wn = 1258; ///< wave number

    const size_t lat = 100;  ///< latitude
    const size_t trc =  11;  ///< truncation
    const size_t fld = 420;  ///< field

    md = new MData( lat, trc, fld );

    test_ = p;

    std::ostringstream fa_name;
    fa_name << "data/antisymmetric_matrix_" << std::setw(5) << std::setfill('0') << wn
            << "_latitude_x_truncation_"    << std::setw(5) << std::setfill('0') << lat
            << "_"                          << std::setw(5) << std::setfill('0') << trc;

    std::ostringstream fb_name;
    fb_name << "data/antisymmetric_matrix_" << std::setw(5) << std::setfill('0') << wn
            << "_truncation_x_field_"       << std::setw(5) << std::setfill('0') << trc
            << "_"                          << std::setw(5) << std::setfill('0') << fld;

    std::ostringstream fc_name;
    fc_name << "data/antisymmetric_matrix_" << std::setw(5) << std::setfill('0') << wn
            << "_latitude_x_field_"         << std::setw(5) << std::setfill('0') << lat
            << "_"                          << std::setw(5) << std::setfill('0') << fld;
    // A

    std::ifstream fa;
    fa.open( fa_name.str().c_str(), ios::in | ios::binary );
    MData::load( md->A , fa, 4 ); /* skip 4 bytes of fortran unformated */
    fa.close();

    // B

    std::ifstream fb;
    fb.open( fb_name.str().c_str(), ios::in | ios::binary );
    MData::load( md->B , fb, 4 ); /* skip 4 bytes of fortran unformated */
    fb.close();

    // Cr

    std::ifstream fc;
    fc.open( fc_name.str().c_str(), ios::in | ios::binary );
    MData::load( md->Cr , fc, 4 ); /* skip 4 bytes of fortran unformated */
    fc.close();

#endif

    size_A = md->A.size1() * md->A.size2();
    size_B = md->B.size1() * md->B.size2();
    size_C = md->C.size1() * md->C.size2();


    std::cout << "A(" << md->m_ << "," << md->k_ << ") * B(" << md->k_ << "," << md->n_ << ")" << std::endl;

    initiate_env();
}

void kronos::Gemm::run()
{
    boost::timer ti;
    boost::timer tc;
    boost::timer tf;

    for( size_t step = 1; step <= steps_hour_ * 24 * forecast_days_; ++step)
    {
        if( step % (steps_hour_ * 24) == 0 )
             std::cout << "> step [" << step << "]" << std::endl;


        // std::cout << "> copying data to devide" << std::endl;

        ti.restart();

        copy_in();

        timers_.copy_in += ti.elapsed();

        //  std::cout << "> computing" << std::endl;

        tc.restart();

        compute();

        timers_.compute += tc.elapsed();

        flops_ += md->flops();

        //  std::cout << "> copying data from devide" << std::endl;

        tf.restart();

        copy_out();

        timers_.copy_out += tf.elapsed();
    }
}

bool kronos::Gemm::verify()
{
    const size_t M = md->m_;
    const size_t N = md->n_;

    const real_t* C  = &md->C.data()[0];
    const real_t* Cr = &md->Cr.data()[0];

    real_t norm = 0.;
    for( size_t i = 0; i < M * N; ++i )
    {
        const real_t diff = C[i] - Cr[i];
        norm += diff*diff;
    }

    norm_L2_ += std::sqrt( norm ) / ( M * N );

//    std::cout << "--- A --------------------------------------------" << std::endl;
//    MData::print( md->A , std::cout, 5, 5 );
//    std::cout << "--------------------------------------------------" << std::endl;

//    std::cout << "--- B --------------------------------------------" << std::endl;
//    MData::print( md->B , std::cout, 5, 5 );
//    std::cout << "--------------------------------------------------" << std::endl;

//    std::cout << "--- C --------------------------------------------" << std::endl;
//    MData::print( md->C , std::cout, 5, 5 );
//    std::cout << "--------------------------------------------------" << std::endl;

//    std::cout << "--- Cr --------------------------------------------" << std::endl;
//    MData::print( md->Cr , std::cout, 5, 5 );
//    std::cout << "--------------------------------------------------" << std::endl;
}

void kronos::Gemm::teardown()
{
    terminate_env();

    delete md;
}

std::string kronos::Gemm::summary()
{
    std::ostringstream ret;

    double sumt = timers_.copy_in + timers_.compute + timers_.copy_out;

    ret << "["  << this->name() << "]\n\n"

        << "L2 Norm : " << norm_L2_ << "\n\n"

        << "timings\n"
        << "\t initiate  : " << std::setw(12) << timers_.copy_in  << " s\n"
        << "\t compute   : " << std::setw(12) << timers_.compute  << " s\n"
        << "\t terminate : " << std::setw(12) << timers_.copy_out << " s\n"
        << "\t sum       : " << std::setw(12) << sumt << " s\n"
        << "quantities\n"
        << "\t flops     : " << std::setw(12) << flops_ << "\n"
        << "\t bytes >   : " << std::setw(12) << copy_into_ / (1024*1024) << " MB \n"
        << "\t bytes <   : " << std::setw(12) << copy_back_ / (1024*1024) << " MB \n"
        << "rates\n";

    if(flops_)
        ret << "\t flops     : " << std::setw(12) << flops_  * 1.0e-9f / timers_.compute << " GFlop/s\n";
    if(flops_)
        ret << "\t flops <>  : " << std::setw(12) << flops_  * 1.0e-9f / sumt << " GFlop/s\n";
    if(copy_into_)
        ret << "\t bytes >   : " << std::setw(12) << copy_into_ / (1024*1024) / timers_.copy_in << " MB/s\n";
    if(copy_back_)
        ret << "\t bytes <   : " << std::setw(12) << copy_back_ / (1024*1024) / timers_.copy_out << " MB/s\n";

    return ret.str();
}

//------------------------------------------------------------------------------------------
