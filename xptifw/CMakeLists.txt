cmake_minimum_required(VERSION 3.20.0)

set(XPTIFW_VERSION 1.0.1)

project (xptifw VERSION "${XPTIFW_VERSION}" LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(XPTIFW_DIR ${CMAKE_CURRENT_LIST_DIR})
set(SAMPLES_DIR ${CMAKE_CURRENT_LIST_DIR}/samples)

# The XPTI framework requires the includes from
# the proxy implementation of XPTI
if (NOT DEFINED XPTI_DIR) # don't overwrite if already set
  if (NOT DEFINED XPTI_SOURCE_DIR)
    # If XPTI_SOURCE_DIR is not provided then assume that xpti is in-tree and not external.
    set(XPTI_DIR ${CMAKE_CURRENT_LIST_DIR}/../xpti)
  else()
    set(XPTI_DIR ${XPTI_SOURCE_DIR}/../xpti)
  endif()
endif()

if (LLVM_ENABLE_PROJECTS)
  # If XPTIFW is built as part of LLVM (i.e. as part of SYCL), then let's
  # re-use existing infrastructure for setting up various hardening flags
  include(HandleLLVMOptions)
endif()

option(XPTI_ENABLE_WERROR OFF)
option(XPTI_BUILD_SAMPLES OFF)
option(XPTI_BUILD_BENCHMARK OFF)
option(XPTI_ENABLE_STATISTICS OFF)
option(XPTI_ENABLE_TESTS ON)

if (XPTI_ENABLE_WERROR)
  if(MSVC)
    set(CMAKE_CXX_FLAGS "/WX ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "-Werror ${CMAKE_CXX_FLAGS}")
  endif()
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_subdirectory(src)

set(xptifw_lib xptifw)
string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lower)
if (WIN32 AND "${build_type_lower}" MATCHES "debug")
  set(xptifw_lib xptifwd)
endif()

add_custom_target(check-xptifw)

if (XPTI_ENABLE_TESTS)
  add_subdirectory(unit_test)
endif()

if (XPTI_BUILD_SAMPLES)
  add_subdirectory(samples/basic_collector)
  add_subdirectory(samples/sycl_perf_collector)
  add_subdirectory(samples/syclur_collector)
  add_subdirectory(basic_test)
endif()

if (XPTI_BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

# Install CMake config files
include(CMakePackageConfigHelpers)
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}${LLVM_LIBDIR_SUFFIX})
set(CONFIG_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/xptifw)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/xptifwConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/xptifwConfig.cmake
  INSTALL_DESTINATION ${CONFIG_INSTALL_DIR}
  PATH_VARS INCLUDE_INSTALL_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/xptifwConfigVersion.cmake
  VERSION ${XPTIFW_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(EXPORT xptifwTargets
  FILE xptifwTargets.cmake
  NAMESPACE xptifw::
  DESTINATION ${CONFIG_INSTALL_DIR}
  COMPONENT xptifw
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/xptifwConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/xptifwConfigVersion.cmake
  DESTINATION ${CONFIG_INSTALL_DIR}
  COMPONENT xptifw
  )

if(NOT WIN32)
  # Install pkg-config file
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/xptifw.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/xptifw.pc
    @ONLY
  )

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/xptifw.pc
    DESTINATION ${LIB_INSTALL_DIR}/pkgconfig
    COMPONENT xptifw
  )
endif()
