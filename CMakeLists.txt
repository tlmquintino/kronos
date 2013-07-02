cmake_minimum_required( VERSION 2.8 )

project( kronos C CXX )

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

### options

option( USE_DOUBLE "enable double" ON )

if( USE_DOUBLE )
	add_definitions( -DUSE_DOUBLE=1 )
	set( CUDA_NVCC_FLAGS "-arch sm_20" )
endif()

### packages

find_package( CUDA )

message( STATUS "CUDA [${CUDA_VERSION_STRING}]" )
if(CUDA_FOUND)
  message( STATUS "  version  [${CUDA_VERSION_STRING}]" )
  message( STATUS "  includes [${CUDA_INCLUDE_DIRS}]" )
  message( STATUS "  libs     [${CUDA_LIBRARIES}]" )
endif()

find_package( OpenCL )

message( STATUS "OPENCL [${OPENCL_FOUND}]" )
if(OPENCL_FOUND)
  message( STATUS "  includes [${OPENCL_INCLUDE_DIRS}]" )
  message( STATUS "  libs     [${OPENCL_LIBRARIES}]" )
endif()

### contents

add_subdirectory( sandbox )

