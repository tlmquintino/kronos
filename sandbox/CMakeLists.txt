if( CUDA_FOUND )
	add_subdirectory( cuda )
endif()

if( OPENCL_FOUND )
	add_subdirectory( opencl )
endif()