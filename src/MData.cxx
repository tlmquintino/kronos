#include <iostream>
#include <iomanip>
#include <limits>

#include <boost/detail/endian.hpp>

#include "MData.h"
#include "Endian.h"
#include "endian.h"

//------------------------------------------------------------------------------------------

void kronos::MData::load( kronos::MData::matrix_t &m,
                          std::istream &in,
                          size_t skip,
                          bool be, bool cm )
{
    real_t b;

    // skip first n bytes
    if(skip)
        in.read( reinterpret_cast<char*>(&b), skip);

    if( cm ) /* column-major ordering */
    {
        for (size_t j = 0; j < m.size2 (); ++j)
            for (size_t i = 0; i < m.size1 (); ++i)
            {
                if( ! in.read( reinterpret_cast<char*>(&b), sizeof(real_t)) )
                    std::cerr << "error reading matrix" << std::endl, ::abort();

#ifdef BOOST_LITTLE_ENDIAN
            if( be )
                kronos::reverse_endian( &b );
#endif

#ifdef BOOST_BIG_ENDIAN
            if( ! be )
                kronos::reverse_endian( &b );
#endif
            m(i, j) = b;

            }
    }
    else /* row-major ordering */
    {
        for (size_t i = 0; i < m.size1 (); ++i)
            for (size_t j = 0; j < m.size2 (); ++j)
            {
                if( ! in.read( reinterpret_cast<char*>(&b), sizeof(real_t)) )
                    std::cerr << "error reading matrix" << std::endl, ::abort();

#ifdef BOOST_LITTLE_ENDIAN
            if( be )
                kronos::reverse_endian( &b );
#endif

#ifdef BOOST_BIG_ENDIAN
            if( ! be )
                kronos::reverse_endian( &b );
#endif
            m(i, j) = b;

            }
    }
}

void kronos::MData::load(kronos::MData::matrix_t &m, const kronos::MData::Block& bm, std::istream &in, size_t skip, bool be, bool cm)
{
    real_t b;

    // skip first n bytes
    if(skip)
        in.read( reinterpret_cast<char*>(&b), skip);

    if( cm ) /* column-major ordering */
    {
        for (size_t j = bm.begin2; j < bm.end2; ++j)
            for (size_t i = bm.begin1; i < bm.end1; ++i)
            {
                if( ! in.read( reinterpret_cast<char*>(&b), sizeof(real_t)) )
                    std::cerr << "error reading matrix" << std::endl, ::abort();

#ifdef BOOST_LITTLE_ENDIAN
            if( be )
                kronos::reverse_endian( &b );
#endif

#ifdef BOOST_BIG_ENDIAN
            if( ! be )
                kronos::reverse_endian( &b );
#endif
            m(i, j) = b;

            }
    }
    else /* row-major ordering */
    {
        for (size_t i = bm.begin1; i < bm.end1; ++i)
            for (size_t j = bm.begin2; j < bm.end2; ++j)
            {
                if( ! in.read( reinterpret_cast<char*>(&b), sizeof(real_t)) )
                    std::cerr << "error reading matrix" << std::endl, ::abort();

#ifdef BOOST_LITTLE_ENDIAN
            if( be )
                kronos::reverse_endian( &b );
#endif

#ifdef BOOST_BIG_ENDIAN
            if( ! be )
                kronos::reverse_endian( &b );
#endif
            m(i, j) = b;

            }
    }
}

void kronos::MData::print(const kronos::MData::matrix_t& m, std::ostream& out, size_t line, size_t col )
{
    if( !line ) line = std::numeric_limits<size_t>::max();
    if( !col  ) col  = std::numeric_limits<size_t>::max();

    size_t i = 0;
    for ( ; i < m.size1() && i < col; ++i )
    {
        size_t j = 0;
        for ( ; j < m.size2() && j < line; ++j )
            out << std::setw(14) << m (i, j) << " ";
        if( j == line )
            out << " ...";
        out << std::endl;
    }

    if( i == col )
        out << " ...";
    out << std::endl;
}

//------------------------------------------------------------------------------------------
