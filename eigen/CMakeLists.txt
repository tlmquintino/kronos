include_directories( ${EIGEN3_INCLUDE_DIR} )

# dense x dense matrix test

add_executable( dense dense.cpp utils.h )

# sparse x dense matrix multiplication

add_executable( sparse sparse.cpp  utils.h )

# memory map a file to a buffer to pass to eigen as dense matrix

add_executable( mmap_dense mmap_dense.cpp  utils.h )

# self-adjoint eigen solver

add_executable( self_adjoint_eigen_solver self_adjoint_eigen_solver.cpp  utils.h )
