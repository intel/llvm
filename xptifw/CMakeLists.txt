cmake_minimum_required(VERSION 3.8)

set(XPTI_VERSION 0.4.1)

project (xptifw VERSION "${XPTI_VERSION}" LANGUAGES CXX)

set(XPTIFW_DIR ${CMAKE_CURRENT_LIST_DIR})
# The XPTI framework requires the includes from
# the proxy implementation of XPTI
set(XPTI_DIR ${CMAKE_CURRENT_LIST_DIR}/../xpti)

# Create a soft option for enabling the use of TBB
option(XPTI_ENABLE_TBB "Enable TBB in the framework" OFF)

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

include_directories(${CMAKE_SOURCE_DIR}/include ${XPTI_DIR}/include)
add_subdirectory(src)
# TODO enable samples and tests back once build problems are resolved.
# Currently builds for unit tests and samples fail mostly with dllimport/dllexport
# mismatch problems:
# syclpi_collector.cpp(32): error C2491: 'xptiTraceInit': definition of dllimport function not allowed
if (0)
add_subdirectory(unit_test)
add_subdirectory(samples/basic_collector)
add_subdirectory(samples/syclpi_collector)
endif()

# The tests in basic_test are written using TBB, so these tests are enabled
# only if TBB has been enabled.
if (0)
  add_subdirectory(basic_test)
endif()
