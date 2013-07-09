#ifndef kronos_MData_h
#define kronos_MData_h

#include "kronos_config.h"

#include <boost/numeric/ublas/matrix.hpp>

namespace kronos {

//------------------------------------------------------------------------------------------

struct MData
{
    typedef boost::numeric::ublas::matrix<real_t> matrix_t;

    MData( size_t m, size_t k, size_t n ) :
        m_(m), k_(k), n_(n),
        A(m,k),
        B(k,n),
        C(m,n),
        Cr(m,n)
    {
    }

    size_t m_;
    size_t k_;
    size_t n_;

    matrix_t A;   ///< A matrix in C = A * B
    matrix_t B;   ///< B matrix in C = A * B
    matrix_t C;   ///< C matrix in C = A * B
    matrix_t Cr;  ///< reference answer

    /// loads a matrix that is stored in big endian, column-major
    static void load_be_cm( matrix_t& m,  std::istream& in );

    /// loads a matrix that is stored in little endian, row-major
    static void load( matrix_t& m,  std::istream& in );

    /// print a matrix in ascii
    static void print( const matrix_t& m,  std::ostream& out, size_t line = 0 );

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
