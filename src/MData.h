#ifndef kronos_MData_h
#define kronos_MData_h

#include "kronos_config.h"

#include <boost/numeric/ublas/matrix.hpp>

namespace kronos {

//------------------------------------------------------------------------------------------

struct MData
{
    typedef boost::numeric::ublas::matrix< real_t, boost::numeric::ublas::column_major, std::vector<real_t> > matrix_t;

    struct Block {
        size_t begin1;
        size_t begin2;
        size_t end1;
        size_t end2;
    };

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

    double flops() { return 2.0 * (double)m_ * (double)k_ * (double)n_; }

    /// loads a matrix from a stream
    /// @param be matrix is encoded in big endian
    /// @param cm matrix is serialized in column-major
    static void load( matrix_t& m, std::istream& in, size_t skip = 0, bool be = true, bool cm = true );

    /// loads a matrix from a stream
    /// @param be matrix is encoded in big endian
    /// @param cm matrix is serialized in column-major
    static void load( matrix_t& m, const Block& b, std::istream& in, size_t skip = 0, bool be = true, bool cm = true );

    /// print a matrix in ascii
    static void print( const matrix_t& m,  std::ostream& out , size_t line = 0, size_t col = 0 );

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
