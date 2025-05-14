# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(UR_LEVEL_ZERO_LOADER_LIBRARY "" CACHE FILEPATH "Path of the Level Zero Loader library")
set(UR_LEVEL_ZERO_INCLUDE_DIR "" CACHE FILEPATH "Directory containing the Level Zero Headers")
set(UR_LEVEL_ZERO_LOADER_REPO "" CACHE STRING "Github repo to get the Level Zero loader sources from")
set(UR_LEVEL_ZERO_LOADER_TAG "" CACHE STRING " GIT tag of the Level Loader taken from github repo")
set(UR_COMPUTE_RUNTIME_REPO "" CACHE STRING "Github repo to get the compute runtime sources from")
set(UR_COMPUTE_RUNTIME_TAG "" CACHE STRING " GIT tag of the compute runtime taken from github repo")

# If UR_COMPUTE_RUNTIME_FETCH_REPO is set to OFF, then UR_COMPUTE_RUNTIME_REPO should be defined and
# should point to the compute runtime repo.
set(UR_COMPUTE_RUNTIME_FETCH_REPO ON CACHE BOOL "Flag to indicate wheather to fetch the compute runtime repo")

# Copy Level Zero loader/headers locally to the build to avoid leaking their path.
set(LEVEL_ZERO_COPY_DIR ${CMAKE_CURRENT_BINARY_DIR}/level_zero_loader)
if (NOT UR_LEVEL_ZERO_LOADER_LIBRARY STREQUAL "")
    get_filename_component(LEVEL_ZERO_LIB_NAME "${UR_LEVEL_ZERO_LOADER_LIBRARY}" NAME)
    set(LEVEL_ZERO_LIBRARY ${LEVEL_ZERO_COPY_DIR}/${LEVEL_ZERO_LIB_NAME})
    message(STATUS "Level Zero Adapter: Copying Level Zero loader to local build tree")
    file(COPY ${UR_LEVEL_ZERO_LOADER_LIBRARY} DESTINATION ${LEVEL_ZERO_COPY_DIR} FOLLOW_SYMLINK_CHAIN)
endif()
if (NOT UR_LEVEL_ZERO_INCLUDE_DIR STREQUAL "")
    set(LEVEL_ZERO_INCLUDE_DIR ${LEVEL_ZERO_COPY_DIR})
    message(STATUS "Level Zero Adapter: Copying Level Zero headers to local build tree")
    file(COPY ${UR_LEVEL_ZERO_INCLUDE_DIR}/ DESTINATION ${LEVEL_ZERO_COPY_DIR})
endif()

if (NOT DEFINED LEVEL_ZERO_LIBRARY OR NOT DEFINED LEVEL_ZERO_INCLUDE_DIR)
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

    if (UR_LEVEL_ZERO_LOADER_REPO STREQUAL "")
        set(UR_LEVEL_ZERO_LOADER_REPO "https://github.com/oneapi-src/level-zero.git")
    endif()
    if (UR_LEVEL_ZERO_LOADER_TAG STREQUAL "")
        set(UR_LEVEL_ZERO_LOADER_TAG c182a1e4fc761f7cddb108df92dc6362c8aea6c0)
    endif()

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

    target_compile_options(ze_loader PRIVATE
        $<$<IN_LIST:$<CXX_COMPILER_ID>,GNU;Clang;Intel;IntelLLVM>:-Wno-error>
        $<$<CXX_COMPILER_ID:MSVC>:/WX- /UUNICODE>
    )

    set(LEVEL_ZERO_LIBRARY ze_loader)
    set(LEVEL_ZERO_INCLUDE_DIR
        ${level-zero-loader_SOURCE_DIR}/include CACHE PATH "Path to Level Zero Headers")
endif()

add_library(LevelZeroLoader INTERFACE)
# The MSVC linker does not like / at the start of a path, so to work around this
# we split it into a link library and a library path, where the path is allowed
# to have leading /.
get_filename_component(LEVEL_ZERO_LIBRARY_SRC "${LEVEL_ZERO_LIBRARY}" DIRECTORY)
get_filename_component(LEVEL_ZERO_LIB_NAME "${LEVEL_ZERO_LIBRARY}" NAME)
target_link_directories(LevelZeroLoader
    INTERFACE "$<BUILD_INTERFACE:${LEVEL_ZERO_LIBRARY_SRC}>"
              "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(LevelZeroLoader
    INTERFACE "${LEVEL_ZERO_LIB_NAME}"
)

file(GLOB LEVEL_ZERO_LOADER_API_HEADERS "${LEVEL_ZERO_INCLUDE_DIR}/*.h")
file(COPY ${LEVEL_ZERO_LOADER_API_HEADERS} DESTINATION ${LEVEL_ZERO_INCLUDE_DIR}/level_zero)
add_library(LevelZeroLoader-Headers INTERFACE)
target_include_directories(LevelZeroLoader-Headers
    INTERFACE "$<BUILD_INTERFACE:${LEVEL_ZERO_INCLUDE_DIR};${LEVEL_ZERO_INCLUDE_DIR}/level_zero>"
              "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

find_path(L0_COMPUTE_RUNTIME_HEADERS
  NAMES "ze_intel_gpu.h"
  PATH_SUFFIXES "level_zero"
)
if(NOT UR_COMPUTE_RUNTIME_REPO AND L0_COMPUTE_RUNTIME_HEADERS)
    set(COMPUTE_RUNTIME_LEVEL_ZERO_INCLUDE "${L0_COMPUTE_RUNTIME_HEADERS}")
    set(COMPUTE_RUNTIME_REPO_PATH "${L0_COMPUTE_RUNTIME_HEADERS}")
elseif (UR_COMPUTE_RUNTIME_FETCH_REPO)
    # Fetch only if UR_COMPUTE_RUNTIME_FETCH_REPO is set to ON.
    if (UR_COMPUTE_RUNTIME_REPO STREQUAL "")
        set(UR_COMPUTE_RUNTIME_REPO "https://github.com/intel/compute-runtime.git")
    endif()
    if (UR_COMPUTE_RUNTIME_TAG STREQUAL "")
        set(UR_COMPUTE_RUNTIME_TAG 25.05.32567.17)
    endif()

    include(FetchContent)
    # Sparse fetch only the dir with level zero headers for experimental features to avoid pulling in the entire compute-runtime.
    FetchContentSparse_Declare(exp-headers ${UR_COMPUTE_RUNTIME_REPO} "${UR_COMPUTE_RUNTIME_TAG}" "level_zero/include")
    FetchContent_GetProperties(exp-headers)
    if(NOT exp-headers_POPULATED)
        FetchContent_Populate(exp-headers)
    endif()

    set(COMPUTE_RUNTIME_LEVEL_ZERO_INCLUDE "${exp-headers_SOURCE_DIR}")
    set(COMPUTE_RUNTIME_REPO_PATH "${exp-headers_SOURCE_DIR}/../..")

# When UR_COMPUTE_RUNTIME_FETCH_REPO is OFF, use UR_COMPUTE_RUNTIME_REPO as repo.
else()

    # Check if UR_COMPUTE_RUNTIME_REPO is set. Throw if not.
    if (UR_COMPUTE_RUNTIME_REPO STREQUAL "")
        message(FATAL_ERROR "UR_COMPUTE_RUNTIME_FETCH_REPO is set to OFF but UR_COMPUTE_RUNTIME_REPO is not set. Please set it to the compute runtime repo.")
    endif()

    set(COMPUTE_RUNTIME_LEVEL_ZERO_INCLUDE "${UR_COMPUTE_RUNTIME_REPO}/level_zero/include")
    set(COMPUTE_RUNTIME_REPO_PATH "${UR_COMPUTE_RUNTIME_REPO}")
endif()

message(STATUS "Using Level Zero include headers from ${COMPUTE_RUNTIME_LEVEL_ZERO_INCLUDE}")

add_library(ComputeRuntimeLevelZero-Headers INTERFACE)
message(STATUS "Level Zero Adapter: Using Level Zero headers from ${COMPUTE_RUNTIME_LEVEL_ZERO_INCLUDE}")
target_include_directories(ComputeRuntimeLevelZero-Headers
    INTERFACE "$<BUILD_INTERFACE:${COMPUTE_RUNTIME_LEVEL_ZERO_INCLUDE}>"
              "$<BUILD_INTERFACE:${COMPUTE_RUNTIME_REPO_PATH}>"
              "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
