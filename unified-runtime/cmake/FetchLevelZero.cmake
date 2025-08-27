# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(TARGET LevelZeroLoader)
  # We only need to run this once.
  return()
endif()

find_package(PkgConfig QUIET)
# LevelZero doesn't install a CMake config target, just PkgConfig,
# so try using that to find the install and if it's not available
# just try to search for the path.
if(PkgConfig_FOUND)
  pkg_check_modules(level-zero level-zero>=1.22.3)
  if(level-zero_FOUND)
    set(LEVEL_ZERO_INCLUDE_DIR "${level-zero_INCLUDEDIR}/level_zero")
    set(LEVEL_ZERO_LIBRARY_SRC "${level-zero_LIBDIR}")
    set(LEVEL_ZERO_LIB_NAME "${level-zero_LIBRARIES}")
    message(STATUS "Level Zero Adapter: Using preinstalled level zero loader at ${level-zero_LINK_LIBRARIES}")
  endif()
else()
  set(L0_HEADER_PATH "loader/ze_loader.h")
  find_path(L0_HEADER ${L0_HEADER_PATH} ${CMAKE_PREFIX_PATH} PATH_SUFFIXES "level_zero")
  find_library(ZE_LOADER NAMES ze_loader HINTS /usr ${CMAKE_PREFIX_PATH})
  if(L0_HEADER AND ZE_LOADER)
    set(LEVEL_ZERO_INCLUDE_DIR "${L0_HEADER}")
    set(LEVEL_ZERO_LIBRARY "${ZE_LOADER}")
    message(STATUS "Level Zero Adapter: Using preinstalled level zero loader at ${LEVEL_ZERO_LIBRARY}")
    add_library(ze_loader INTERFACE)
  endif()
endif()

if(NOT LEVEL_ZERO_LIB_NAME AND NOT LEVEL_ZERO_LIBRARY)
  message(STATUS "Level Zero Adapter: Download Level Zero loader and headers from github.com")

  # Workaround warnings/errors for Level Zero build
  set(CMAKE_CXX_FLAGS_BAK "${CMAKE_CXX_FLAGS}")
  if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-truncation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-extra-semi")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
  endif()
  set(BUILD_STATIC ON)

  set(UR_LEVEL_ZERO_LOADER_REPO "https://github.com/oneapi-src/level-zero.git")
  # Remember to update the pkg_check_modules minimum version above when updating the
  # clone tag
  set(UR_LEVEL_ZERO_LOADER_TAG 35c037cdf4aa9a2e6df34b6f1ce1bdc86ac5422f)

  # Disable due to a bug https://github.com/oneapi-src/level-zero/issues/104
  set(CMAKE_INCLUDE_CURRENT_DIR OFF)
  # Prevent L0 loader from exporting extra symbols
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)

  set(CMAKE_MSVC_RUNTIME_LIBRARY_BAK "${CMAKE_MSVC_RUNTIME_LIBRARY}")
  # UMF has not yet been able to build as static
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  message(STATUS "Level Zero Adapter: Will fetch Level Zero Loader from ${UR_LEVEL_ZERO_LOADER_REPO}")
  include(FetchContent)
  FetchContent_Declare(level-zero-loader
    GIT_REPOSITORY    ${UR_LEVEL_ZERO_LOADER_REPO}
    GIT_TAG           ${UR_LEVEL_ZERO_LOADER_TAG}
    )
  if(MSVC)
    set(USE_Z7 ON)
  endif()
  FetchContent_MakeAvailable(level-zero-loader)
  FetchContent_GetProperties(level-zero-loader)

  # Restore original flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BAK}")
  set(CMAKE_MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY_BAK}")
  set(LEVEL_ZERO_LIBRARY ze_loader)
  set(LEVEL_ZERO_INCLUDE_DIR
    ${level-zero-loader_SOURCE_DIR}/include CACHE PATH "Path to Level Zero Headers")

  set(LEVEL_ZERO_TARGET_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")

  file(GLOB LEVEL_ZERO_LOADER_API_HEADERS "${LEVEL_ZERO_INCLUDE_DIR}/*.h")
  file(COPY ${LEVEL_ZERO_LOADER_API_HEADERS} DESTINATION ${LEVEL_ZERO_INCLUDE_DIR}/level_zero)

  target_compile_options(ze_loader PRIVATE
    $<$<IN_LIST:$<CXX_COMPILER_ID>,GNU;Clang;Intel;IntelLLVM>:-Wno-error>
    $<$<CXX_COMPILER_ID:MSVC>:/WX- /UUNICODE>
    )
endif()

add_library(LevelZeroLoader INTERFACE)
# The MSVC linker does not like / at the start of a path, so to work around this
# we split it into a link library and a library path, where the path is allowed
# to have leading /.
if(NOT LEVEL_ZERO_LIBRARY_SRC OR NOT LEVEL_ZERO_LIB_NAME)
get_filename_component(LEVEL_ZERO_LIBRARY_SRC "${LEVEL_ZERO_LIBRARY}" DIRECTORY)
get_filename_component(LEVEL_ZERO_LIB_NAME "${LEVEL_ZERO_LIBRARY}" NAME)
endif()

if(NOT LEVEL_ZERO_TARGET_INCLUDE_DIR)
  set(LEVEL_ZERO_TARGET_INCLUDE_DIR ${LEVEL_ZERO_INCLUDE_DIR})
  endif()

target_link_directories(LevelZeroLoader
    INTERFACE "$<BUILD_INTERFACE:${LEVEL_ZERO_LIBRARY_SRC}>"
              "$<INSTALL_INTERFACE:${LEVEL_ZERO_TARGET_INCLUDE_DIR}>"
)
target_link_libraries(LevelZeroLoader
    INTERFACE "${LEVEL_ZERO_LIB_NAME}"
)

add_library(LevelZeroLoader-Headers INTERFACE)
target_include_directories(LevelZeroLoader-Headers
    INTERFACE "$<BUILD_INTERFACE:${LEVEL_ZERO_INCLUDE_DIR}>"
              "$<INSTALL_INTERFACE:${LEVEL_ZERO_TARGET_INCLUDE_DIR}>"
)
find_path(L0_COMPUTE_RUNTIME_HEADERS
  NAMES "ze_intel_gpu.h"
  PATH_SUFFIXES "level_zero"
)
if(L0_COMPUTE_RUNTIME_HEADERS)
    set(COMPUTE_RUNTIME_LEVEL_ZERO_INCLUDE "${L0_COMPUTE_RUNTIME_HEADERS}")
    set(COMPUTE_RUNTIME_REPO_PATH "${L0_COMPUTE_RUNTIME_HEADERS}")
else()
    set(UR_COMPUTE_RUNTIME_REPO "https://github.com/intel/compute-runtime.git")
    set(UR_COMPUTE_RUNTIME_TAG 25.05.32567.17)

    include(FetchContent)
    # Sparse fetch only the dir with level zero headers for experimental features to avoid pulling in the entire compute-runtime.
    FetchContentSparse_Declare(exp-headers ${UR_COMPUTE_RUNTIME_REPO} "${UR_COMPUTE_RUNTIME_TAG}" "level_zero/include")
    FetchContent_GetProperties(exp-headers)
    if(NOT exp-headers_POPULATED)
      FetchContent_Populate(exp-headers)
    endif()
    set(COMPUTE_RUNTIME_LEVEL_ZERO_INCLUDE "${exp-headers_SOURCE_DIR}")
    set(COMPUTE_RUNTIME_REPO_PATH "${exp-headers_SOURCE_DIR}/../..")
endif()

message(STATUS "Using Level Zero include headers from ${COMPUTE_RUNTIME_LEVEL_ZERO_INCLUDE}")

add_library(ComputeRuntimeLevelZero-Headers INTERFACE)
message(STATUS "Level Zero Adapter: Using Level Zero headers from ${COMPUTE_RUNTIME_LEVEL_ZERO_INCLUDE}")
target_include_directories(ComputeRuntimeLevelZero-Headers
    INTERFACE "$<BUILD_INTERFACE:${COMPUTE_RUNTIME_LEVEL_ZERO_INCLUDE}>"
              "$<BUILD_INTERFACE:${COMPUTE_RUNTIME_REPO_PATH}>"
              "$<INSTALL_INTERFACE:${LEVEL_ZERO_TARGET_INCLUDE_DIR}>"
)

set(LEVEL_ZERO_INCLUDE_DIR "${LEVEL_ZERO_INCLUDE_DIR}" CACHE PATH INTERNAL)
