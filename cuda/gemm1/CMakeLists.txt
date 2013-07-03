include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

cuda_add_executable( gemm1 matrix_sizes.h matrix_mult_kernel.h matrix_mult.cu )
