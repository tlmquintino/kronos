# TODO:
#   * cuda streamming / async

cmake_minimum_required( VERSION 2.8 )

project( kronos C CXX )

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

### packages

# set( Boost_DEBUG 1 )
set(Boost_USE_MULTITHREADED    ON )
if( DEFINED BUILD_SHARED_LIBS AND BUILD_SHARED_LIBS )
    set(Boost_USE_STATIC_LIBS  OFF )
else()
    set(Boost_USE_STATIC_LIBS  ON )
endif()
find_package( Boost COMPONENTS filesystem regex program_options date_time thread system  )

find_package( MKL )

find_package( CUDA )

find_package( OpenCL )

if( OPENCL_FOUND )
	find_package( ViennaCL )
endif()

find_package( BLAS )

### options

option( SANDBOX "build sandbox" OFF )

option( USE_DOUBLE "enable double" ON )
if( USE_DOUBLE )
    message( STATUS "real_t --> double" )
    set( USE_DOUBLE 1 )
    set( CUDA_NVCC_FLAGS "-arch sm_20" ) # should check first if compute capability is ok
else()
    message( STATUS "real_t --> float" )
    set( USE_DOUBLE 0 )
endif()

### configuratio files

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )
configure_file( kronos_config.h.in kronos_config.h @ONLY )

### contents

add_subdirectory( src )

if( SANDBOX AND CUDA_FOUND )
    add_subdirectory( cuda )
endif()

if( SANDBOX AND OPENCL_FOUND )
    add_subdirectory( opencl )
endif()

### summary

message( STATUS "CUDA [${CUDA_VERSION_STRING}]" )
if(CUDA_FOUND)
  message( STATUS "  version  [${CUDA_VERSION_STRING}]" )
  message( STATUS "  includes [${CUDA_INCLUDE_DIRS}]" )
  message( STATUS "  libs     [${CUDA_LIBRARIES}]" )
endif()

message( STATUS "OpenCL [${OPENCL_FOUND}]" )
if(OPENCL_FOUND)
  message( STATUS "  includes [${OPENCL_INCLUDE_DIRS}]" )
  message( STATUS "  libs     [${OPENCL_LIBRARIES}]" )
endif()

message( STATUS "MKL [${MKL_FOUND}]" )
if(MKL_FOUND)
  message( STATUS "  includes [${MKL_INCLUDE_DIRS}]" )
  message( STATUS "  libs     [${MKL_LIBRARIES}]" )
endif()

message( STATUS "ViennaCL [${ViennaCL_FOUND}]" )
if(ViennaCL_FOUND)
  message( STATUS "  includes [${VIENNACL_INCLUDE_DIRS}]" )
  message( STATUS "  libs     [${VIENNACL_LIBRARIES}]" )
endif()

message( STATUS "BLAS [${BLAS_FOUND}]" )
if(BLAS_FOUND)
  message( STATUS "  libs     [${BLAS_LIBRARIES}]" )
endif()

