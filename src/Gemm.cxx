#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <numeric>

#include <unistd.h>

#include "boost/date_time/posix_time/posix_time.hpp"

#include "kronos_config.h"

#include "Gemm.h"
#include "Endian.h"
#include "MData.h"

using namespace std;

//------------------------------------------------------------------------------------------

void random_init(real_t* data, int size)
{
    for ( size_t i = 0; i < size; ++i )
        data[i] = rand() / (real_t) RAND_MAX;
}

//------------------------------------------------------------------------------------------

kronos::Gemm::Gemm() :
    norm_L2_(0.),
    copy_into_(0.),
    copy_back_(0.),
    flops_(0.),
    steps_(1),
    align_to_(0),
    threads_(1),
    print_results_(false)
{
    timers_.copy_in  = 0.;
    timers_.compute  = 0.;
    timers_.copy_out = 0.;
}

void kronos::Gemm::setup( const boost::filesystem::path& p,
                          const size_t wn,
                          const size_t lat,
                          const size_t trc,
                          const std::vector<size_t>& fields )
{
    timers_.copy_in  = 0.;
    timers_.compute  = 0.;
    timers_.copy_out = 0.;

    norm_L2_   = 0.;
    copy_into_ = 0.,
    copy_back_ = 0.,
    flops_     = 0.;

    wn_   = wn;
    lat_  = lat;
    trc_  = trc;
    sumf_ = std::accumulate(fields.begin(),fields.end(),0);

    char hname [255];
    if( ::gethostname( hname, 255 ) ) std::cout << "error in gethostname()" << std::endl, ::exit(-1);

#define USE_SMALL_MATRICES

#ifdef USE_SMALL_MATRICES

    lat_  = 4*1024;
    trc_  = 4*1024;
    sumf_ = 6*1024;

    md = new MData( lat_, trc_, sumf_, align_to_ );

    size_A = md->A.size1() * md->A.size2();
    size_B = md->B.size1() * md->B.size2();
    size_C = md->C.size1() * md->C.size2();

    mem_size_A = sizeof(real_t) * size_A;
    mem_size_B = sizeof(real_t) * size_B;
    mem_size_C = sizeof(real_t) * size_C;

    MData::matrix_t& A  = md->A;
    MData::matrix_t& B  = md->B;
    MData::matrix_t& Cr = md->Cr;

    random_init( &A.data()[0], size_A );
    random_init( &B.data()[0], size_B );

//    Cr = prod( A , B );

#ifdef INIT_MAT
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

#else

    md = new MData( lat, trc, sumf_, align_to_ );

    size_A = md->A.size1() * md->A.size2();
    size_B = md->B.size1() * md->B.size2();
    size_C = md->C.size1() * md->C.size2();

    mem_size_A = sizeof(real_t) * size_A;
    mem_size_B = sizeof(real_t) * size_B;
    mem_size_C = sizeof(real_t) * size_C;

    // load A

    std::ostringstream fa_name;
    fa_name << p.string()
            << "/antisymmetric_matrix_" << std::setw(5) << std::setfill('0') << wn
            << "_latitude_x_truncation_"<< std::setw(5) << std::setfill('0') << lat
            << "_"                      << std::setw(5) << std::setfill('0') << trc;

    MData::Block ba;
    ba.begin1 = 0;
    ba.end1   = md->m_;
    ba.begin2 = 0;
    ba.end2   = md->k_;

    std::ifstream fa;
    fa.open( fa_name.str().c_str(), ios::in | ios::binary );
    MData::load( md->A, ba, fa, 4 ); /* skip 4 bytes of fortran unformated */
    fa.close();

    // loop over fields

    MData::Block bb;
    MData::Block bc;

    bb.begin1 = 0;
    bb.end1   = md->k_;
    bb.begin2 = 0;
    bb.end2   = 0;

    bc.begin1 = 0;
    bc.end1   = md->m_;
    bc.begin2 = 0;
    bc.end2   = 0;

    for( size_t f = 0; f < fields.size(); ++f )
    {
        size_t fld = fields[f];

//        std::cout << fld << std::endl;

        // load series of trc x field matrics to form B

        bb.begin2  = bb.end2;
        bb.end2   += fld;

//        std::cout << "reading B(" << bb.begin1 << ":" << bb.end1 << "," << bb.begin2 << ":" << bb.end2 << ")" << std::endl;

        std::ostringstream fb_name;
        fb_name << p.string()
                << "/antisymmetric_matrix_" << std::setw(5) << std::setfill('0') << wn
                << "_truncation_x_field_"   << std::setw(5) << std::setfill('0') << trc
                << "_"                      << std::setw(5) << std::setfill('0') << fld;

        std::ifstream fb;
        fb.open( fb_name.str().c_str(), ios::in | ios::binary );
        MData::load( md->B, bb, fb, 4 ); /* skip 4 bytes of fortran unformated */
        fb.close();

        // load series of lat x field matrics to form Cr

        bc.begin2  = bc.end2;
        bc.end2   += fld;

//        std::cout << "reading C(" << bc.begin1 << ":" << bc.end1 << "," << bc.begin2 << ":" << bc.end2 << ")" << std::endl;

        std::ostringstream fc_name;
        fc_name << p.string()
                << "/antisymmetric_matrix_" << std::setw(5) << std::setfill('0') << wn
                << "_latitude_x_field_"     << std::setw(5) << std::setfill('0') << lat
                << "_"                      << std::setw(5) << std::setfill('0') << fld;

        std::ifstream fc;
        fc.open( fc_name.str().c_str(), ios::in | ios::binary );
        MData::load( md->Cr, bc, fc, 4 ); /* skip 4 bytes of fortran unformated */
        fc.close();
    }

#endif

    initiate_env();

    pre_process();
}

void kronos::Gemm::run()
{
    boost::posix_time::ptime t1;
    boost::posix_time::time_duration dt;

    for( size_t step = 1; step <= steps_; ++step)
    {
       if( step % ( 6 ) == 0 )
            std::cout << "> step [" << step << "]" << std::endl;

//         std::cout << "> copying data to devide" << std::endl;

        t1 = boost::posix_time::microsec_clock::universal_time();

        copy_in();

        dt = boost::posix_time::microsec_clock::universal_time() - t1;
        timers_.copy_in += dt.total_microseconds() / 1E6; /* in seconds */

//          std::cout << "> computing" << std::endl;

        t1 = boost::posix_time::microsec_clock::universal_time();

        compute();

        dt = boost::posix_time::microsec_clock::universal_time() - t1;
        timers_.compute += dt.total_microseconds() / 1E6; /* in seconds */

        std::cout << "gflop/s " << std::setw(12) << md->flops()  * 1.0e-9f / ( dt.total_microseconds() / 1E6 )  << std::endl;

        flops_ += md->flops();

//          std::cout << "> copying data from devide" << std::endl;

        t1 = boost::posix_time::microsec_clock::universal_time();

        copy_out();

        dt = boost::posix_time::microsec_clock::universal_time() - t1;
        timers_.copy_out += dt.total_microseconds() / 1E6; /* in seconds */
    }
}

bool kronos::Gemm::verify()
{
    post_process();

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

    if( print_results_ )
    {
        const size_t pr = 7;

        std::cout << "--- A --------------------------------------------" << std::endl;
        MData::print( md->A , std::cout, pr, pr );
        std::cout << "--------------------------------------------------" << std::endl;

        std::cout << "--- B --------------------------------------------" << std::endl;
        MData::print( md->B , std::cout, pr, pr );
        std::cout << "--------------------------------------------------" << std::endl;

        std::cout << "--- C --------------------------------------------" << std::endl;
        MData::print( md->C , std::cout, pr, pr );
        std::cout << "--------------------------------------------------" << std::endl;

        std::cout << "--- Cr -------------------------------------------" << std::endl;
        MData::print( md->Cr , std::cout, pr, pr );
        std::cout << "--------------------------------------------------" << std::endl;
    }

   return true;
}

void kronos::Gemm::teardown()
{
    terminate_env();

    delete md;
}

string kronos::Gemm::prologue()
{
    std::ostringstream ret;

    ret       << "wn, "   << std::setw(5) << wn_ << ", "
              << "lat, "  << std::setw(5) << lat_ << ", "
              << "trc, "  << std::setw(5) << trc_ << ", "
              << "flds, " << std::setw(5) << sumf_ << ", " ;

    return ret.str();
}

std::string kronos::Gemm::summary()
{
    std::ostringstream ret;

    double sumt = timers_.copy_in + timers_.compute + timers_.copy_out;

    ret << "L2, " << std::setw(12) << norm_L2_ << ", " << std::flush;

    //--------------

    ret << "gflop/s, "   << std::setw(12) << flops_  * 1.0e-9f / timers_.compute << ", ";

    ret << "gflop/s<>, " << std::setw(12) << flops_  * 1.0e-9f / sumt << ", ";

    if(copy_into_)
        ret << ">MB/s, " << std::setw(12) << copy_into_ / (1024*1024) / timers_.copy_in<< ", ";
    else
        ret << ">MB/s, " << std::setw(12) << copy_into_<< ", ";

    if(copy_back_)
        ret << "<MB/s, " << std::setw(12) << copy_back_ / (1024*1024) / timers_.copy_out<< ", ";
    else
        ret << "<MB/s, " << std::setw(12) << copy_back_<< ", ";

    //--------------

    ret << "flops, "   << std::setw(12) << flops_ << ", ";
    ret << "B>, " << std::setw(12) << copy_into_ << ", ";
    ret << "B<, " << std::setw(12) << copy_back_ << ", ";

    ret << "t>, " << std::setw(12) << timers_.copy_in << ", ";
    ret << "tc, " << std::setw(12) << timers_.compute << ", ";
    ret << "t<, " << std::setw(12) << timers_.copy_out << ", ";

    return ret.str();
}

void kronos::Gemm::align_to(const size_t& value)
{
    align_to_ = value;
}

void kronos::Gemm::steps(const size_t& steps)
{
    steps_ = steps;
}

void kronos::Gemm::threads(const size_t& threads)
{
    threads_ = threads;
}

//------------------------------------------------------------------------------------------
