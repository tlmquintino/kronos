#include <vector>
#include <algorithm> 
#include <iostream>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include<sys/types.h>
#include<sys/msg.h>
#include<sys/shm.h>

double myrand()
{
    // returns double between 0.0 and 1
    return (double)rand() / (double)RAND_MAX;
}

int main()
{
    const int input_point_count = 1000;

    // Generate some input values...
    std::vector<double> input_values;
    input_values.reserve(input_point_count);
    std::generate_n(std::back_inserter(input_values), input_point_count, myrand);

    // create input dense matrix
    Eigen::MatrixXd A(input_point_count, 1);
    A.col(0) = Eigen::VectorXd::Map(&input_values[0], input_point_count);

    // create output dense matrix
    const int output_point_count = 90;
    Eigen::MatrixXd B(1, output_point_count);

    //
    // TODO: look in shared memory and load, otherwise generate
    //
    
    // generate the i,j coordinates that we want in our sparse matrix
    // (create random values here)
    std::vector<Eigen::Triplet<double> > insertions;
    for (unsigned int i = 0; i < 500; i++)
    {
        int row = (int)(output_point_count * myrand());
        int col = (int)(input_point_count * myrand());
        double val = myrand();

        insertions.push_back(Eigen::Triplet<double>(row, col, val));
    }

    // create a sparse matrix and fill with the (i,j,value) items
    Eigen::SparseMatrix<double> weights = Eigen::SparseMatrix<double>(output_point_count, input_point_count);
    weights.setFromTriplets(insertions.begin(), insertions.end());

    // one-line transformation coming up
    B = weights * A;

    // write the output values to another vector
    std::vector<double> output_values(output_point_count);
    Eigen::Map<Eigen::VectorXd>(output_values.data(), B.rows()) = B.col(0);

    // done
    return 0;
 
                
}
