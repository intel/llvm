cmake_minimum_required(VERSION 3.8)

set(XPTI_VERSION 0.6.0)

project (xptifw VERSION "${XPTI_VERSION}" LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(XPTIFW_DIR ${CMAKE_CURRENT_LIST_DIR})

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
if (SYCL_USE_LIBCXX)
  if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR
    (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    if ((NOT (DEFINED SYCL_LIBCXX_INCLUDE_PATH)) OR (NOT (DEFINED SYCL_LIBCXX_LIBRARY_PATH)))
      message(FATAL_ERROR "When building with libc++ SYCL_LIBCXX_INCLUDE_PATHS and"
                          "SYCL_LIBCXX_LIBRARY_PATH should be set")
    endif()
    set(CMAKE_CXX_FLAGS "-nostdinc++ -I ${SYCL_LIBCXX_INCLUDE_PATH} ${CMAKE_CXX_FLAGS}")
  else()
    message(FATAL_ERROR "Build with libc++ is not yet supported for this compiler")
  endif()
endif()

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_subdirectory(src)

add_custom_target(check-xptifw)

if (XPTI_ENABLE_TESTS)
  add_subdirectory(unit_test)
endif()

if (XPTI_BUILD_SAMPLES)
  add_subdirectory(samples/basic_collector)
  add_subdirectory(samples/syclpi_collector)
  add_subdirectory(basic_test)
endif()

if (XPTI_BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()
