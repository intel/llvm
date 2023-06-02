# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(SYCL_L0_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(L0_DIR "${SYCL_L0_DIR}/sycl/plugins/unified_runtime/ur/adapters/level_zero")

# fetch xpti proxy library for the tracing layer
FetchSource(https://github.com/intel/llvm.git sycl-nightly/20230601 "sycl/plugins/unified_runtime/ur" ${SYCL_L0_DIR})

set(TARGET_NAME ur_adapter_level_zero)

if (NOT DEFINED LEVEL_ZERO_LIBRARY OR NOT DEFINED LEVEL_ZERO_INCLUDE_DIR)
    message(STATUS "Download Level Zero loader and headers from github.com")

    set(LEVEL_ZERO_LOADER_REPO "https://github.com/oneapi-src/level-zero.git")
    set(LEVEL_ZERO_LOADER_TAG v1.8.8)

    # Disable due to a bug https://github.com/oneapi-src/level-zero/issues/104
    set(CMAKE_INCLUDE_CURRENT_DIR OFF)
    # Prevent L0 loader from exporting extra symbols
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)

    message(STATUS "Will fetch Level Zero Loader from ${LEVEL_ZERO_LOADER_REPO}")
    include(FetchContent)
    FetchContent_Declare(level-zero-loader
        GIT_REPOSITORY    ${LEVEL_ZERO_LOADER_REPO}
        GIT_TAG           ${LEVEL_ZERO_LOADER_TAG}
    )
    if (WIN32)
        set(USE_Z7 ON)
    endif()
    FetchContent_MakeAvailable(level-zero-loader)
    FetchContent_GetProperties(level-zero-loader)

    set(LEVEL_ZERO_LIBRARY ze_loader)
    set(LEVEL_ZERO_INCLUDE_DIR
        ${level-zero-loader_SOURCE_DIR}/include CACHE PATH "Path to Level Zero Headers")
endif()

add_library (LevelZeroLoader INTERFACE)
# The MSVC linker does not like / at the start of a path, so to work around this
# we split it into a link library and a library path, where the path is allowed
# to have leading /.
get_filename_component(LEVEL_ZERO_LIBRARY_SRC "${LEVEL_ZERO_LIBRARY}" DIRECTORY)
get_filename_component(LEVEL_ZERO_LIB_NAME "${LEVEL_ZERO_LIBRARY}" NAME)
target_link_directories(LevelZeroLoader
    INTERFACE "${LEVEL_ZERO_LIBRARY_SRC}"
)
target_link_libraries(LevelZeroLoader
    INTERFACE "${LEVEL_ZERO_LIB_NAME}"
)
if (WIN32)
    # TODO: fix building level-zero loader on Windows
    # target_compile_options(${LEVEL_ZERO_LIB_NAME} /UUNICODE)
else()
    target_compile_options(${LEVEL_ZERO_LIB_NAME} PUBLIC
        -Wno-unused-but-set-variable
        -Wno-pedantic
        -Wno-unused-parameter
        -Wno-error
    )
endif()

add_library (LevelZeroLoader-Headers INTERFACE)
target_include_directories(LevelZeroLoader-Headers
    INTERFACE "${LEVEL_ZERO_INCLUDE_DIR}"
)

add_library(${TARGET_NAME}
    SHARED
    ${L0_DIR}/ur_loader_interface.cpp
    ${L0_DIR}/ur_level_zero_common.hpp
    ${L0_DIR}/ur_level_zero_context.hpp
    ${L0_DIR}/ur_level_zero_device.hpp
    ${L0_DIR}/ur_level_zero_event.hpp
    ${L0_DIR}/ur_level_zero_mem.hpp
    ${L0_DIR}/ur_level_zero_kernel.hpp
    ${L0_DIR}/ur_level_zero_platform.hpp
    ${L0_DIR}/ur_level_zero_program.hpp
    ${L0_DIR}/ur_level_zero_queue.hpp
    ${L0_DIR}/ur_level_zero_sampler.hpp
    ${L0_DIR}/ur_level_zero.cpp
    ${L0_DIR}/ur_level_zero_common.cpp
    ${L0_DIR}/ur_level_zero_context.cpp
    ${L0_DIR}/ur_level_zero_device.cpp
    ${L0_DIR}/ur_level_zero_event.cpp
    ${L0_DIR}/ur_level_zero_mem.cpp
    ${L0_DIR}/ur_level_zero_kernel.cpp
    ${L0_DIR}/ur_level_zero_platform.cpp
    ${L0_DIR}/ur_level_zero_program.cpp
    ${L0_DIR}/ur_level_zero_queue.cpp
    ${L0_DIR}/ur_level_zero_sampler.cpp
    ${L0_DIR}/../../ur.cpp
    ${L0_DIR}/../../usm_allocator_config.hpp
)

set_target_properties(${TARGET_NAME} PROPERTIES
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
)

target_link_libraries(${TARGET_NAME} PRIVATE
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::common
    LevelZeroLoader-Headers
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${L0_DIR}/../../../
    LevelZeroLoader-Headers
)

if(UNIX)
    set(GCC_COVERAGE_COMPILE_FLAGS "-fvisibility=hidden -fvisibility-inlines-hidden -fno-strict-aliasing")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
endif()

