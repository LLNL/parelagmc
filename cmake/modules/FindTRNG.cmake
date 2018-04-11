# Defines the following variables:
#   - TRNG_FOUND
#   - TRNG_LIBRARY
#   - TRNG_INCLUDE_DIRS

# Find the header files
find_path(TRNG_INCLUDE_DIRS trng/ 
  HINTS ${TRNG_DIR} ${CMAKE_BINARY_DIR}/external/trng4 $ENV{TRNG_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with TRNG header.")
find_path(TRNG_INCLUDE_DIRS trng/)

# Find the library
find_library(TRNG_LIBRARY trng4 
  HINTS ${TRNG_DIR} ${CMAKE_BINARY_DIR}/external/trng4 $ENV{TRNG_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The TRNG library.")
find_library(TRNG_LIBRARY trng4)

# If a trng installation is found
if (TRNG_INCLUDE_DIRS AND TRNG_LIBRARY)
    # Setup the imported target
    if (NOT TARGET TRNG::trng)
      # Check if we have shared or static libraries
      include(ParELAGMC_CMakeUtilities)
      parelagmc_determine_library_type(${TRNG_LIBRARY} TRNG_LIB_TYPE)

      add_library(TRNG::trng ${TRNG_LIB_TYPE} IMPORTED)
    endif (NOT TARGET TRNG::trng)

    # Set library 
    set_property(TARGET TRNG::trng 
      PROPERTY IMPORTED_LOCATION ${TRNG_LIBRARY})

    # Add include path
    set_property(TARGET TRNG::trng APPEND
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${TRNG_INCLUDE_DIRS})
        
    # Set the include directories
    set(TRNG_INCLUDE_DIRS ${TRNG_INCLUDE_DIRS}
      CACHE PATH
      "Directories in which to find headers for TRNG.")
    mark_as_advanced(FORCE TRNG_INCLUDE_DIRS)

    # Set the libraries
    set(TRNG_LIBRARIES TRNG::trng)
    mark_as_advanced(FORCE TRNG_LIBRARY)

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(TRNG
      DEFAULT_MSG
      TRNG_LIBRARY TRNG_INCLUDE_DIRS)
else()
    
    # If installation not found, automatically download and install
    include(ExternalProject)

    set(STAGE_DIR           ${CMAKE_BINARY_DIR}/stage)
    set(TRNG_URL            https://www.numbercrunch.de/trng/trng-4.19.tar.gz)
    set(TRNG_FILE           trng-4.19.tar.gz)
    set(TRNG_SOURCE_DIR     ${STAGE_DIR}/trng-4.19)

    if(TRNG_INSTALL_PREFIX)
        set(TRNG_INSTALL_DIR ${TRNG_INSTALL_PREFIX})
    elseif(DEFINED ENV{TRNG_INSTALL_PREFIX})
        set(TRNG_INSTALL_DIR $ENV{TRNG_INSTALL_PREFIX})
    else()
        set(TRNG_INSTALL_DIR ${CMAKE_BINARY_DIR}/external/trng4)
        message(STATUS "TRNG will be installed in ${TRNG_INSTALL_DIR}.")
    endif()

    ExternalProject_Add(
        trng 
        #DOWNLOAD_COMMAND    wget ${TRNG_URL}   
        #PATCH_COMMAND       tar -xvf ../${TRNG_FILE} && mv trng-4.19 ${TRNG_SOURCE_DIR}
        URL                 ${TRNG_URL}
        SOURCE_DIR          ${TRNG_SOURCE_DIR}
        CONFIGURE_COMMAND   ${TRNG_SOURCE_DIR}/configure --prefix=${TRNG_INSTALL_DIR}
        BUILD_COMMAND       $(MAKE)
        INSTALL_COMMAND     $(MAKE) install 
    )
  
    set(TRNG_INCLUDE_DIRS "")
    list(APPEND TRNG_INCLUDE_DIRS
        ${TRNG_INSTALL_DIR}/include
    ) 

    set(TRNG_LIBRARY ${TRNG_INSTALL_DIR}/lib/libtrng4.a)
    # Setup the imported target
    if (NOT TARGET TRNG::trng)
        add_library(TRNG::trng STATIC IMPORTED) 
    endif (NOT TARGET TRNG::trng)

    # Set library
    set_property(TARGET TRNG::trng
      PROPERTY IMPORTED_LOCATION ${TRNG_LIBRARY})

    # Set the libraries
    set(TRNG_LIBRARIES TRNG::trng)
    mark_as_advanced(FORCE TRNG_LIBRARY)

endif()

                                                                 
