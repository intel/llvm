# Copyright (C) 2022-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if (UR_BUILD_ADAPTER_L0 OR UR_BUILD_ADAPTER_L0_V2)
    include(FetchLevelZero)
    set(UMF_BUILD_LEVEL_ZERO_PROVIDER ON CACHE INTERNAL "Build Level Zero Provider")
    set(UMF_LEVEL_ZERO_INCLUDE_DIR "${LEVEL_ZERO_INCLUDE_DIR}" CACHE INTERNAL "Level Zero headers")
else()
    set(UMF_BUILD_LEVEL_ZERO_PROVIDER OFF CACHE INTERNAL "Build Level Zero Provider")
endif()

if (UR_BUILD_ADAPTER_CUDA)
    find_package(CUDAToolkit 10.1 REQUIRED)
    set(UMF_BUILD_CUDA_PROVIDER ON CACHE INTERNAL "Build UMF CUDA provider")
    set(UMF_CUDA_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}" CACHE INTERNAL "CUDA headers")
else()
    set(UMF_BUILD_CUDA_PROVIDER OFF CACHE INTERNAL "Build UMF CUDA provider")
endif()

add_ur_library(ur_common STATIC
    ur_util.cpp
    ur_util.hpp
    latency_tracker.hpp
    offload_bundle_parser.cpp
    offload_bundle_parser.hpp
    $<$<PLATFORM_ID:Windows>:windows/ur_lib_loader.cpp>
    $<$<PLATFORM_ID:Linux,Darwin>:linux/ur_lib_loader.cpp>
)

add_library(${PROJECT_NAME}::common ALIAS ur_common)

target_include_directories(ur_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

if (UR_STATIC_ADAPTER_L0)
    if (UMF_BUILD_SHARED_LIBRARY)
        message(STATUS "Static adapter is not compatible with shared UMF, switching to fully statically linked UMF")
        set(UMF_BUILD_SHARED_LIBRARY OFF)
    endif()
endif()
  
set(UR_USE_EXTERNAL_UMF ON CACHE BOOL "Use a pre-built UMF if available")

if(UR_USE_EXTERNAL_UMF)
  # v1.1.0-dev2 is not strictly required and they both are compatible
  find_package(umf 1.0.0 QUIET)
endif()
if(umf_FOUND)
  message(STATUS "Using preinstalled UMF at ${umf_DIR}, ignoring UMF build related options")
  # Add an alias matching the FetchContent case
  add_library(umf::headers ALIAS umf::umf_headers)
else()
  set(UMF_REPO "https://github.com/oneapi-src/unified-memory-framework.git")

  # commit 503d302a72f719a3f11fce0e610f07a3793549d9
  # Author: Łukasz Stolarczuk <lukasz.stolarczuk@intel.com>
  # Date:   Mon Aug 18 11:14:25 2025 +0200
  # Merge pull request #1496 from sarnex/gtest
  # [CMake] Support preinstalled GTest
  set(UMF_TAG v1.1.0-dev2)

  if(NOT FETCHCONTENT_SOURCE_DIR_UNIFIED-MEMORY-FRAMEWORK)
    message(STATUS "Will fetch Unified Memory Framework from ${UMF_REPO}")
  endif()

  include(FetchContent)
  FetchContent_Declare(unified-memory-framework
    GIT_REPOSITORY    ${UMF_REPO}
    GIT_TAG           ${UMF_TAG}
    )
  set(UMF_BUILD_TESTS OFF CACHE INTERNAL "Build UMF tests")
  set(UMF_BUILD_EXAMPLES OFF CACHE INTERNAL "Build UMF examples")
  set(UMF_BUILD_SHARED_LIBRARY ${UMF_BUILD_SHARED_LIBRARY} CACHE INTERNAL "Build UMF shared library")

  FetchContent_MakeAvailable(unified-memory-framework)
  FetchContent_GetProperties(unified-memory-framework)
endif()

if(UR_ENABLE_LATENCY_HISTOGRAM)
  find_package(hdr_histogram QUIET)
  if(hdr_histogram_FOUND)
    set(hdr_histogram_SOURCE_DIR "${hdr_histogram_DIR}")
  else()
    set(HDR_HISTOGRAM_BUILD_STATIC CACHE INTERNAL ON "")
    set(HDR_HISTOGRAM_BUILD_SHARED CACHE INTERNAL OFF "")

    include(FetchContent)
    FetchContent_Declare(hdr_histogram
        GIT_REPOSITORY    https://github.com/HdrHistogram/HdrHistogram_c.git
        GIT_TAG           0.11.8
    )

    FetchContent_MakeAvailable(hdr_histogram)
    FetchContent_GetProperties(hdr_histogram)
  endif()
  target_link_libraries(ur_common PUBLIC hdr_histogram_static)
  target_include_directories(ur_common PUBLIC $<BUILD_INTERFACE:${hdr_histogram_SOURCE_DIR}/include>)
  target_compile_options(ur_common PUBLIC -DUR_ENABLE_LATENCY_HISTOGRAM=1)
endif()

target_link_libraries(ur_common PUBLIC
    ${CMAKE_DL_LIBS}
    ${PROJECT_NAME}::headers
)

if (UNIX)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    target_link_libraries(ur_common PUBLIC Threads::Threads)
endif()

install(TARGETS ur_common
    EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

add_library(ur_umf INTERFACE)
target_sources(ur_umf INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/umf_helpers.hpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/umf_pools/disjoint_pool_config_parser.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ur_pool_manager.hpp>
)

add_library(${PROJECT_NAME}::umf ALIAS ur_umf)

target_link_libraries(ur_umf INTERFACE
    umf::umf
    umf::headers
)
