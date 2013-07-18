if( Boost_FOUND )

    include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
    include_directories( ${Boost_INCLUDE_DIRS} )

    add_library( gemmcore Gemm.cxx Gemm.h CpuGemm.cxx  CpuGemm.h MData.h MData.cxx Endian.h )

    add_executable( gemm main.cxx )

    target_link_libraries( gemm gemmcore ${Boost_LIBRARIES} )

    if( CUDA_FOUND )
        cuda_add_library( cudagemm CudaUtils.h CudaGemm.cxx CudaGemm.h CublasGemm.cxx CublasGemm.h cuda_gemm.h cuda_gemm.cu )
        cuda_add_cublas_to_target( cudagemm )
        target_link_libraries( gemm cudagemm )
    endif()

    if( OPENCL_FOUND )
         add_library( clgemm CLGemm.cxx CLGemm.h kronos_opencl.h kronos_opencl.cxx )
         target_link_libraries( gemm clgemm )
    endif()

    if( MKL_FOUND )
        include_directories( ${MKL_INCLUDE_DIRS} )
        add_library( mklgemm MKLGemm.cxx MKLGemm.h )
        target_link_libraries( mklgemm ${MKL_LIBRARIES} )
        target_link_libraries( gemm mklgemm )
    endif()

    if( ViennaCL_FOUND )
        include_directories( ${VIENNACL_INCLUDE_DIRS} )
        add_library( viennaclgemm ViennaCLGemm.cxx ViennaCLGemm.h )
        target_link_libraries( viennaclgemm ${VIENNACL_LIBRARIES} )
        target_link_libraries( gemm viennaclgemm )
    endif()

    if( BLAS_FOUND )
        add_library( blasgemm BlasGemm.cxx BlasGemm.h )
        target_link_libraries( blasgemm ${BLAS_LIBRARIES} )
        target_link_libraries( gemm blasgemm )
    endif()

endif()
