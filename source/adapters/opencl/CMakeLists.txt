# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(OPENCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "OpenCL adapter directory")

set(UR_OPENCL_INCLUDE_DIR "" CACHE PATH "Directory containing the OpenCL Headers")
set(UR_OPENCL_ICD_LOADER_LIBRARY "" CACHE FILEPATH "Path of the OpenCL ICD Loader library")

find_package(Threads REQUIRED)

set(TARGET_NAME ur_adapter_opencl)

add_ur_adapter(${TARGET_NAME} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/ur_interface_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_buffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/context.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sampler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/usm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/usm_p2p.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/virtual_mem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.hpp
)

set_target_properties(${TARGET_NAME} PROPERTIES
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
)

if(UR_OPENCL_INCLUDE_DIR)
    set(OpenCLIncludeDirectory ${UR_OPENCL_INCLUDE_DIR})
else()
    FetchContent_Declare(OpenCL-Headers
        GIT_REPOSITORY  "https://github.com/KhronosGroup/OpenCL-Headers.git"
        GIT_TAG         main
    )
    FetchContent_MakeAvailable(OpenCL-Headers)
    FetchContent_GetProperties(OpenCL-Headers
        SOURCE_DIR OpenCLIncludeDirectory
    )
endif()

# The OpenCL target can be set manually on upstream cmake to avoid using
# find_package().
if(UR_OPENCL_ICD_LOADER_LIBRARY)
    set(OpenCLICDLoaderLibrary ${UR_OPENCL_ICD_LOADER_LIBRARY})
else()
    find_package(OpenCL 3.0)
    if(NOT OpenCL_FOUND)
        FetchContent_Declare(OpenCL-ICD-Loader
            GIT_REPOSITORY  "https://github.com/KhronosGroup/OpenCL-ICD-Loader.git"
            GIT_TAG         main
        )
        FetchContent_MakeAvailable(OpenCL-ICD-Loader)
    endif()
    set(OpenCLICDLoaderLibrary OpenCL::OpenCL)
endif()

message(STATUS "OpenCL Include Directory: ${OpenCLIncludeDirectory}")
message(STATUS "OpenCL ICD Loader Library: ${OpenCLICDLoaderLibrary}")

# Suppress a compiler message about undefined CL_TARGET_OPENCL_VERSION.
# Define all symbols up to OpenCL 3.0.
target_compile_definitions(ur_adapter_opencl PRIVATE
    CL_TARGET_OPENCL_VERSION=300
    CL_USE_DEPRECATED_OPENCL_1_2_APIS
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${OpenCLIncludeDirectory}
    "${CMAKE_CURRENT_SOURCE_DIR}/../../"
)

target_link_libraries(${TARGET_NAME} PRIVATE
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::unified_malloc_framework
    Threads::Threads
    ${OpenCLICDLoaderLibrary}
)
