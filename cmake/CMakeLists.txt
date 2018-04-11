# Build the CMake configuration files
include(CMakePackageConfigHelpers)

set(MODULES_TO_INSTALL
  modules/CorrectWindowsPaths.cmake
  modules/FindMOONOLITH.cmake
  modules/FindTRNG.cmake
  modules/FindPackageMultipass.cmake
  modules/ParELAGMC_CMakeUtilities.cmake
  )

# Create the version for the build tree
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}"
  COMPATIBILITY SameMajorVersion)

# Create the build-tree configuration file
set(INCLUDE_INSTALL_DIRS "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/src")
set(LIB_INSTALL_DIR "${PROJECT_BINARY_DIR}/src")
set(MODULE_INSTALL_DIR "${PROJECT_SOURCE_DIR}/cmake/modules")

configure_package_config_file(ParELAGMCConfig.cmake.in
  ${CMAKE_BINARY_DIR}/ParELAGMCConfig.cmake
  INSTALL_DESTINATION ${CMAKE_BINARY_DIR}
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR MODULE_INSTALL_DIR)

#
# Install tree stuff
#

# Create the install-tree configuration file
set(INCLUDE_INSTALL_DIRS include/parelagmc)
set(LIB_INSTALL_DIR lib)
set(CMAKE_INSTALL_DIR lib/cmake/parelagmc)
set(MODULE_INSTALL_DIR ${CMAKE_INSTALL_DIR}/modules)

configure_package_config_file(
  ParELAGMCConfig.cmake.in
  ParELAGMCConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_DIR}
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR MODULE_INSTALL_DIR)

# Install the version file
install(
  FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${CMAKE_INSTALL_DIR})

# Install the modules
install(FILES ${MODULES_TO_INSTALL} DESTINATION ${MODULE_INSTALL_DIR})
