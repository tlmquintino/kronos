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
    flops_(0.)
{
    timers_.copy_in  = 0.;
    timers_.compute  = 0.;
    timers_.copy_out = 0.;
}

void kronos::Gemm::setup(const boost::filesystem::path& p)
{

#if 1

    const size_t lat = 2; ///< latitude
    const size_t trc = 2; ///< truncation
    const size_t fld = 2; ///< field

    mm_ = new MData( lat, trc, fld );

    MData::matrix_t& A  = mm_->A;
    MData::matrix_t& B  = mm_->B;
    MData::matrix_t& Cr = mm_->Cr;

    A(0,0) = 2;
    A(1,0) = 3;
    A(0,1) = 2;
    A(1,1) = 3;

    B(0,0) = 4;
    B(1,0) = 5;
    B(0,1) = 4;
    B(1,1) = 5;

    Cr(0,0) = 18;
    Cr(1,0) = 27;
    Cr(0,1) = 18;
    Cr(1,1) = 27;

#else
    const size_t  wn = 639; ///< wave number

    const size_t lat = 442; ///< latitude
    const size_t trc = 321; ///< truncation
    const size_t fld = 420; ///< field

    mm_ = new MData( lat, trc, fld );

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
    MData::load( mm_->A , fa, 4 ); /* skip 4 bytes of fortran unformated */
    fa.close();

    // B

    std::ifstream fb;
    fb.open( fb_name.str().c_str(), ios::in | ios::binary );
    MData::load( mm_->B , fb, 4 ); /* skip 4 bytes of fortran unformated */
    fb.close();

    // Cr

    std::ifstream fc;
    fc.open( fc_name.str().c_str(), ios::in | ios::binary );
    MData::load( mm_->Cr , fc, 4 ); /* skip 4 bytes of fortran unformated */
    fc.close();

#endif

    std::cout << "A(" << mm_->m_ << "," << mm_->k_ << ") * B(" << mm_->k_ << "," << mm_->n_ << ")" << std::endl;

}

void kronos::Gemm::run()
{
    // init

    boost::timer ti;

    this->initiate();

    timers_.copy_in += ti.elapsed();

    // compute

    boost::timer tr;

    this->compute();

    timers_.compute += tr.elapsed();

    flops_ += mm_->flops();

    // term

    boost::timer tf;

    this->terminate();

    timers_.copy_out += tf.elapsed();

}

bool kronos::Gemm::verify()
{
    const size_t M = mm_->m_;
    const size_t N = mm_->n_;

    const real_t* C  = &mm_->C.data()[0];
    const real_t* Cr = &mm_->Cr.data()[0];

    real_t norm = 0.;
    for( size_t i = 0; i < M * N; ++i )
    {
        const real_t diff = C[i] - Cr[i];
        norm += diff*diff;
    }

    norm_L2_ += std::sqrt( norm ) / ( M * N );

    std::cout << "--- A --------------------------------------------" << std::endl;
    MData::print( mm_->A , std::cout, 5, 5 );
    std::cout << "--------------------------------------------------" << std::endl;

    std::cout << "--- B --------------------------------------------" << std::endl;
    MData::print( mm_->B , std::cout, 5, 5 );
    std::cout << "--------------------------------------------------" << std::endl;

    std::cout << "--- C --------------------------------------------" << std::endl;
    MData::print( mm_->C , std::cout, 5, 5 );
    std::cout << "--------------------------------------------------" << std::endl;

    std::cout << "--- Cr --------------------------------------------" << std::endl;
    MData::print( mm_->Cr , std::cout, 5, 5 );
    std::cout << "--------------------------------------------------" << std::endl;
}

void kronos::Gemm::teardown()
{
    delete mm_;
}

std::string kronos::Gemm::summary()
{
    std::ostringstream ret;

    ret << "["  << this->name() << "]\n\n"

        << "L2 Norm : " << norm_L2_ << "\n\n"

        << "timings\n"
        << "\t initiate  : " << timers_.copy_in  << " s\n"
        << "\t compute   : " << timers_.compute  << " s\n"
        << "\t terminate : " << timers_.copy_out << " s\n"
        << "quantities\n"
        << "\t flops     : " << flops_ << "\n"
        << "\t bytes >   : " << copy_into_ / (1024*1024) << " MB \n"
        << "\t bytes <   : " << copy_back_ / (1024*1024) << " MB \n"
        << "rates\n";
    if(flops_)
        ret << "\t flops     : " << flops_  * 1.0e-9f / timers_.compute << " GFlop/s\n";
    if(copy_into_)
        ret << "\t bytes >   : " << copy_into_ / (1024*1024) / timers_.copy_in << " MB/s\n";
    if(copy_back_)
        ret << "\t bytes >   : " << copy_back_ / (1024*1024) / timers_.copy_out << " MB/s\n";

    return ret.str();
}

//------------------------------------------------------------------------------------------
