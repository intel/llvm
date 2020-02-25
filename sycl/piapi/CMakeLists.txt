cmake_minimum_required(VERSION 3.4)
project(piapi VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(ExternalProject)
find_package(Threads REQUIRED)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Options 

# TODO: Not possible to turn this off at the moment
option(PI_OPENCL_HEADERS "Should OpenCL headers be included" ON)

option(PI_BUILD_CUDA "Whether to build the pi_cuda plugin" OFF)
option(PI_BUILD_LEVEL_ZERO "Whether to build the pi_level_zero plugin" OFF)

set(PI_LIBDIR_SUFFIX "" CACHE STRING "Library suffix when installing")
set(PI_XPTI_INCLUDE_DIR "" CACHE STRING "Location of the XPTI include directory")

################################################################################
# Main library

set(pi_include_dir "${PROJECT_SOURCE_DIR}/include")
set(pi_source_dir "${PROJECT_SOURCE_DIR}/src")
set(pi_library_base_dir "${CMAKE_CURRENT_BINARY_DIR}")
set(pi_library_output_dir "${pi_library_base_dir}/lib${PI_LIBDIR_SUFFIX}")
set(pi_binary_output_dir "${pi_library_base_dir}/bin")

set(CONF_INCLUDE_DIRS "")

add_library(piapi STATIC
  "${pi_include_dir}/pi/pi.h"
  "${pi_include_dir}/pi/pi.hpp"
  "${pi_source_dir}/pi.cpp"
  "$<$<PLATFORM_ID:Windows>:${pi_source_dir}/windows_pi.cpp>"
  "$<$<OR:$<PLATFORM_ID:Linux>,$<PLATFORM_ID:Darwin>>:${pi_source_dir}/posix_pi.cpp>"
)
set_target_properties(piapi PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${pi_library_output_dir}"
)
list(APPEND CONF_INCLUDE_DIRS "${pi_include_dir}")
list(APPEND CONF_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(piapi PUBLIC
  "$<BUILD_INTERFACE:${pi_include_dir}>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
)
target_include_directories(piapi INTERFACE
  "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(piapi PUBLIC ${CMAKE_DL_LIBS})
target_link_libraries(piapi PUBLIC Threads::Threads)
target_compile_definitions(piapi PUBLIC CL_TARGET_OPENCL_VERSION=210)

################################################################################
# Export header

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# Define export header for the plugins
# We don't have the plugins yet, so we rely on a shim shared library
set(pi_export_header "${CMAKE_CURRENT_BINARY_DIR}/piapi_export.h")
add_library(pi_export_library_shim EXCLUDE_FROM_ALL SHARED "${pi_source_dir}/shim.cpp")
include(GenerateExportHeader)
generate_export_header(pi_export_library_shim
  BASE_NAME PIAPI
  EXPORT_MACRO_NAME PIAPI_EXPORT
  EXPORT_FILE_NAME "${pi_export_header}"
  STATIC_DEFINE PIAPI_STATIC
)

################################################################################
# OpenCL

if(PI_OPENCL_HEADERS)
  find_package(OpenCL REQUIRED)

  # Do not link against the OpenCL::OpenCL library, just need the headers
  target_include_directories(piapi PUBLIC
    "${OpenCL_INCLUDE_DIR}"
  )
endif()

################################################################################
# XPTI tracing

if(PI_XPTI_INCLUDE_DIR)
  target_compile_definitions(piapi PRIVATE XPTI_ENABLE_INSTRUMENTATION=1)
  target_include_directories(piapi PUBLIC "${PI_XPTI_INCLUDE_DIR}")
endif()

################################################################################
# Install

# Export the package for use from the build-tree
export(PACKAGE piapi)

install(DIRECTORY "${pi_include_dir}"
  DESTINATION . COMPONENT piapi
)
install(TARGETS piapi
  EXPORT piapiTargets
  ARCHIVE DESTINATION "lib${PI_LIBDIR_SUFFIX}" COMPONENT piapi
  RUNTIME DESTINATION "bin" COMPONENT piapi
)

################################################################################
# Interface library that simplifies exporting symbols for the plugins

add_library(pi_export_library INTERFACE)
target_include_directories(pi_export_library INTERFACE "${pi_library_output_dir}")
target_compile_definitions(pi_export_library INTERFACE
  pi_export_library_shim_EXPORTS
  PI_REQUIRES_EXPORT
)
target_sources(pi_export_library INTERFACE
  "${pi_source_dir}/plugin_pi_hooks.cpp"
)

################################################################################
# Tests

enable_testing()

################################################################################
# Plugins

add_subdirectory(plugins)

################################################################################
# Package

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/piapiConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY ExactVersion
)

configure_file(piapiConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/piapiConfig.cmake
  COPYONLY
)

set(config_package_location lib/cmake/piapi)
install(EXPORT piapiTargets
  FILE piapiTargets.cmake
  NAMESPACE piapi::
  DESTINATION ${config_package_location}
  COMPONENT piapi
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/piapiConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/piapiConfigVersion.cmake
  DESTINATION ${config_package_location}
  COMPONENT piapi
)
