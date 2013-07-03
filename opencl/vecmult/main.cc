#include <iostream>

#include "kronoscl.h"

int main(int argc, char** argv)
{
    std::cout << "KRONOS" << std::endl;

    kronos::CL k;
    
    k.load( "vecmult.cl" );

    k.init();
	
    k.run();
}
