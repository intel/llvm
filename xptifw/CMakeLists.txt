cmake_minimum_required(VERSION 2.8.9)

set(XPTI_VERSION 0.4.1)
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

project (xptifw)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

include_directories(${CMAKE_SOURCE_DIR}/include ${XPTI_DIR}/include)
add_subdirectory(src)
add_subdirectory(unit_test)
add_subdirectory(samples/basic_collector)
add_subdirectory(samples/syclpi_collector)
# The tests in basic_test are written using TBB, so these tests are enabled
# only if TBB has been enabled.
if (XPTI_ENABLE_TBB)
  add_subdirectory(basic_test)
endif()
