# - Try to find OpenCL
# Once done this will define
#
#  MKL_FOUND         - system has Intel MKL
#  MKL_INCLUDE_DIRS  - the MKL include directories
#  MKL_LIBRARIES     - link these to use MKL

if( $ENV{MKL_ROOT} )
	list( APPEND __MKL_PATHS $ENV{MKL_ROOT} )
endif()

if( MKL_ROOT )
	list( APPEND __MKL_PATHS ${MKL_ROOT} )
endif()

find_path(MKL_INCLUDE_DIR mkl.h PATHS ${__MKL_PATHS} PATH_SUFFIXES include NO_DEFAULT_PATH)
find_path(MKL_INCLUDE_DIR mkl.h PATHS ${__MKL_PATHS} PATH_SUFFIXES include )

if( MKL_INCLUDE_DIR ) # use include dir to find libs

    set( MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR} )

	if( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
	  get_filename_component( MKL_LIB_PATH ${MKL_INCLUDE_DIR}/../lib/intel64 ABSOLUTE )
	  set( __libsfx _lp64 )
	else()
	  get_filename_component( MKL_LIB_PATH ${MKL_INCLUDE_DIR}/../lib/ia32 ABSOLUTE )
	  set( __libsfx "" )
	endif()

    find_library( MKL_LIB_INTEL      NAMES mkl_intel${__libsfx} PATHS ${MKL_LIB_PATH} )
    find_library( MKL_LIB_SEQUENTIAL NAMES mkl_sequential PATHS ${MKL_LIB_PATH} )
    find_library( MKL_LIB_CORE       NAMES mkl_core PATHS ${MKL_LIB_PATH} )

    if( MKL_LIB_INTEL AND MKL_LIB_SEQUENTIAL AND MKL_LIB_CORE )
        set( MKL_LIBRARIES ${MKL_LIB_INTEL} ${MKL_LIB_SEQUENTIAL} ${MKL_LIB_CORE} )
    endif()

endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args( MKL DEFAULT_MSG 
                                   MKL_LIBRARIES MKL_INCLUDE_DIRS )

mark_as_advanced( MKL_INCLUDE_DIR MKL_LIB_LAPACK MKL_LIB_INTEL MKL_LIB_SEQUENTIAL MKL_LIB_CORE )

