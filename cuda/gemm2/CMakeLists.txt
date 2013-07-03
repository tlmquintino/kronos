if( Boost_FOUND )

	include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

	cuda_add_executable( gemm2 main.cxx matrix_sizes.h matrix_mult.h matrix_mult.cu )

	target_link_libraries( gemm2 ${Boost_LIBRARIES} )

endif()