#include <iostream>

#include "kronoscl.h"

int main(int argc, char** argv)
{
    std::cout << "KRONOS" << std::endl;

    kronos::CL k;
    
    if( sizeof(kronos::CL::data_t) == sizeof(float) )
        k.load( "fvecmult.cl" );

    if( sizeof(kronos::CL::data_t) == sizeof(double) )
        k.load( "dvecmult.cl" );

    k.init();
	
    k.run();
}
