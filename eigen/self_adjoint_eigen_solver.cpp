#include <iostream>
#include <sstream>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "utils.h"

using namespace Eigen;

int main()
{
    std::ostringstream osi;
    std::ostringstream osf;

    MatrixXd X = MatrixXd::Random(5,5);
    MatrixXd A = X + X.transpose();
    
    osi << A << std::endl;

    SelfAdjointEigenSolver<MatrixXd> es(A);

//    std::cout << es.eigenvalues() << std::endl;
//    std::cout << es.eigenvectors() << std::endl;

    MatrixXd D = es.eigenvalues().asDiagonal();
    MatrixXd V = es.eigenvectors();

    osf << V * D * V.inverse() << std::endl;

    if( osi.str() == osf.str() )
    {
        std::cout << "OK" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "FAIL" << std::endl;
        return -1;
    }

    return 0;
}
