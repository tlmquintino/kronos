if( Boost_FOUND )

    if( MKL_FOUND )
       include_directories( ${MKL_INCLUDE_DIRS} )
    endif()

    include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

    cuda_add_executable( gemm3 main.cxx matrix_sizes.h matrix_mult.h matrix_mult.cu mmcublas.h mmcublas.cxx )

    cuda_add_cublas_to_target( gemm3 )

    target_link_libraries( gemm3 ${Boost_LIBRARIES} )

    if( MKL_FOUND )
       target_link_libraries( gemm3 ${MKL_LIBRARIES} )
    endif()

endif()
