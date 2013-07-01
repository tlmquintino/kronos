cmake_minimum_required( VERSION 2.8 )

project( kronos C CXX )

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

find_package( OpenCL )

if( NOT OPENCL_FOUND )
	message( FATAL_ERROR "OpenCL not found" )
endif()

include_directories( ${OPENCL_INCLUDE_DIRS} )

# provide runtime path to load cl files
add_definitions( -DKRONOS_USE_DOUBLE=1 )
add_definitions( -DKRONOS_SRC_DIR="${kronos_SOURCE_DIR}" )

#  core library

list( APPEND kronos_core_files kronoscl.cc kronoscl.h util.cc util.h )

add_library( kronos_core ${kronos_core_files})

#  executable

add_executable( kronos main.cc )

target_link_libraries(  kronos kronos_core ${OPENCL_LIBRARIES} )

add_custom_target( kronos_cls SOURCES vecmult.cl )
