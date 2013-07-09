#include <iostream>
#include <iomanip>
#include <limits>

#include <boost/detail/endian.hpp>

#include "MData.h"
#include "Endian.h"
#include "endian.h"

//------------------------------------------------------------------------------------------

void kronos::MData::load_be_cm(kronos::MData::matrix_t &m, std::istream &in)
{
    real_t b;

    // skip first 8 bytes
    in.read( reinterpret_cast<char*>(&b), sizeof(real_t));

    for (size_t j = 0; j < m.size2 (); ++j)
        for (size_t i = 0; i < m.size1 (); ++i)
        {
            if( ! in.read( reinterpret_cast<char*>(&b), sizeof(real_t)) )
            {
                std::cerr << "error reading matrix" << std::endl;
                ::abort();
            }

//            std::cerr << "m(" << i << "," << j << ") "  << b << std::endl;

#ifdef BOOST_LITTLE_ENDIAN
            kronos::reverse_endian( &b );
#endif

//            std::cout << "m(" << i << "," << j << ") " << b << std::endl;

            m(i, j) = b;
        }
}

void kronos::MData::load( kronos::MData::matrix_t& m, std::istream &in )
{
    for (size_t i = 0; i < m.size1 (); ++i)
        for (size_t j = 0; j < m.size2 (); ++j)
            in >> m (i, j);
}

void kronos::MData::print(const kronos::MData::matrix_t& m, std::ostream& out, size_t line)
{
    if( !line ) line = std::numeric_limits<size_t>::max();

    for ( size_t i = 0; i < m.size1 (); ++i )
    {
        size_t j = 0;
        for ( ; j < m.size2() && j < line; ++j )
            out << std::setw(14) << m (i, j) << " ";
        if( j == line )
            out << " ...";
        out << std::endl;
    }
}

//------------------------------------------------------------------------------------------
