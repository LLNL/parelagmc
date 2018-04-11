
# Search in user defined path or in path from cmake install
find_path(ParMoonolith_INSTALLATION_PATH
    NAME config/moonolith_config.cmake
    HINTS ${ParMoonolith_DIR} ${CMAKE_BINARY_DIR}/external/par_moonolith
    $ENV{ParMoonolith_DIR}
    )

if(ParMoonolith_INSTALLATION_PATH)
    message(STATUS "Found ParMoonolith installation at ${ParMoonolith_INSTALLATION_PATH}")
    include(${ParMoonolith_INSTALLATION_PATH}/config/moonolith_config.cmake)
    include(FindPackageHandleStandardArgs)

    find_package_handle_standard_args(ParMoonolith
      REQUIRED_VARS MOONOLITH_LIBRARIES MOONOLITH_INCLUDES
    )

    mark_as_advanced(MOONOLITH_INCLUDES MOONOLITH_LIBRARIES)

    if(ParMoonolith_FOUND)
        add_custom_target(par_moonolith)
    endif()
endif()

if(NOT ParMoonolith_FOUND)
	#Automatically download
	include(ExternalProject)

	set(STAGE_DIR 				    "${CMAKE_BINARY_DIR}/stage")
	set(ParMoonolith_URL 			https://bitbucket.org/zulianp/par_moonolith.git)
	set(ParMoonolith_SOURCE_DIR 	${STAGE_DIR}/par_moonolith)
	#set(ParMoonolith_BIN_DIR 		${STAGE_DIR}/par_moonolith/bin)

	if(ParMoonolith_INSTALL_PREFIX)
		set(ParMoonolith_INSTALL_DIR ${ParMoonolith_INSTALL_PREFIX})
	elseif(DEFINED ENV{ParMoonolith_INSTALL_PREFIX})
		set(ParMoonolith_INSTALL_DIR $ENV{ParMoonolith_INSTALL_PREFIX})
	else()
		set(ParMoonolith_INSTALL_DIR ${CMAKE_BINARY_DIR}/external/par_moonolith)
		message(STATUS "ParMoonolith will be installed in ${ParMoonolith_INSTALL_DIR}.")
	endif()


	list(APPEND ParMoonolith_CMAKE_ARGS
		"-DCMAKE_INSTALL_PREFIX=${ParMoonolith_INSTALL_DIR}"
		"-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
		"-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
		)

	ExternalProject_Add(
		par_moonolith 
		UPDATE_COMMAND		"" 
		PREFIX              ${STAGE_DIR}
		GIT_REPOSITORY 		${ParMoonolith_URL}
		DOWNLOAD_DIR 		${STAGE_DIR} 
		INSTALL_DIR         ${ParMoonolith_INSTALL_DIR}
		CMAKE_ARGS 			"${ParMoonolith_CMAKE_ARGS}"
		LOG_CONFIGURE		1
		LOG_BUILD 			1
	)

	list(APPEND MOONOLITH_INCLUDES 
		${ParMoonolith_INSTALL_DIR}/include
		${ParMoonolith_INSTALL_DIR}/include/kernels
		)

	set(MOONOLITH_LIBRARIES "")
	list(APPEND MOONOLITH_LIBRARIES 
		"-L${ParMoonolith_INSTALL_DIR}/lib"
		"-lmoonolith_opencl"
		"-lpar_moonolith"
		"-lpar_moonolith_intersection"
		"-lpar_moonolith_mpi"
		"-lpar_moonolith_tree"
		"-lpar_moonolith_utils"
		)

	set(ParMoonolith_FOUND TRUE)

endif()
