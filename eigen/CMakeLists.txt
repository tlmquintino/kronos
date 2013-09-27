include_directories( ${EIGEN3_INCLUDE_DIR} )

# dense x dense matrix test

add_executable( dense dense.cpp )

# sparse x dense matrix multiplication

add_executable( sparse sparse.cpp )

# memory map a file to a buffer to pass to eigen as dense matrix

add_executable( mmap_dense mmap_dense.cpp )
