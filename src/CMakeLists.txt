if( Boost_FOUND )

    include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

    add_library( gemmcore Gemm.cxx Gemm.h CpuGemm.cxx  CpuGemm.h )

    add_executable( gemm main.cxx )

    target_link_libraries( gemm gemmcore ${Boost_LIBRARIES} )

    if( CUDA_FOUND )
        cuda_add_library( cudagemm CudaGemm.cxx CudaGemm.h CublasGemm.cxx CublasGemm.h )
        cuda_add_cublas_to_target( cudagemm )
        target_link_libraries( gemm cudagemm )
    endif()

    if( OPENCL_FOUND )
         add_library( clgemm CLGemm.cxx CLGemm.h )
         target_link_libraries( gemm clgemm )
    endif()

    if( MKL_FOUND )
        include_directories( ${MKL_INCLUDE_DIRS} )
        add_library( mklgemm MKLGemm.cxx MKLGemm.h )
        target_link_libraries( mklgemm ${MKL_LIBRARIES} )
        target_link_libraries( gemm mklgemm )
    endif()

endif()
