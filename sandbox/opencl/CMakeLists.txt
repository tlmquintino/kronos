include_directories( ${OPENCL_INCLUDE_DIRS} )

add_subdirectory( vecmult )
add_subdirectory( vecsum )

add_subdirectory( gemm1 )
add_subdirectory( gemm2 )
add_subdirectory( gemm3 )
add_subdirectory( gemm4 )
add_subdirectory( gemm5 )
add_subdirectory( gemm6 )
add_subdirectory( gemm8 )
