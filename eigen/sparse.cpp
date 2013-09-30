#include <vector>
#include <algorithm> 
#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "utils.h"

using namespace Eigen;

//-----------------------------------------------------------------------------------------

template<typename Scalar, typename Index = size_t >
class Tri
{
public:
  Tri() : m_row(0), m_col(0), m_value(0) {}

  Tri(const Index& i, const Index& j, const Scalar& v = Scalar(0))
    : m_row(i), m_col(j), m_value(v)
  {}

  /** \returns the row index of the element */
  const Index& row() const { return m_row; }
  void row( const Index& ) const { return m_row; }

  /** \returns the column index of the element */
  const Index& col() const { return m_col; }

  /** \returns the value of the element */
  const Scalar& value() const { return m_value; }

  Index m_row, m_col;
  Scalar m_value;

};

//-----------------------------------------------------------------------------------------

typedef std::vector< Tri<double> > Coef_t;

//-----------------------------------------------------------------------------------------

int main()
{
    const size_t n = 10;
    const size_t k = 10;
    const size_t m = 20;

    // build coefficients of sparse matrix

    Coef_t coefs;
    coefs.resize( 3*n );
    size_t i = 0;
    Coef_t::iterator it = coefs.begin();
    for( ; it != coefs.end(); ++it, ++i )
    {
        const size_t r = std::min( i / 3 , n-1 );
        const size_t c = std::min( i / 3 + i % 3 , k-1 );
        it->m_row   = r ;
        it->m_col   = c;
        it->m_value = i+1;
    }

//    for( size_t i = 0; i < coefs.size(); ++i )
//        std::cout << "row " << coefs[i].row()  << " col " << coefs[i].col() << " value " << coefs[i].value() << std::endl;

    // assemble sparse matrix

    SparseMatrix<double> A(n,k);
    A.setFromTriplets(coefs.begin(),coefs.end());

    std::cout << sep << A << sep << std::endl;

    // make dense matrix

    MatrixXd B( k, m);

    init_matrix(B);

    // compute result

    MatrixXd C( n, m);

    C = A * B;

    std::cout << sep << C.format(CleanFmt) << sep << std::endl;

    return 0;
}
