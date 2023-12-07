# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(OPENCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "OpenCL adapter directory")

set(TARGET_NAME ur_adapter_opencl)

add_ur_adapter(${TARGET_NAME}
        SHARED
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
        ${CMAKE_CURRENT_SOURCE_DIR}/platform.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/platform.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/program.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/queue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sampler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/usm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/usm_p2p.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.hpp
)

set_target_properties(${TARGET_NAME} PROPERTIES
        VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
        SOVERSION "${PROJECT_VERSION_MAJOR}"
)

find_package(Threads REQUIRED)

# The OpenCL target can be set manually on upstream cmake to avoid using find_package().
if (NOT UR_OPENCL_ICD_LOADER_LIBRARY)
set(OCL_HEADERS_REPO
"https://github.com/intel-innersource/applications.compilers.source.opencl-headers")
set(OCL_LOADER_REPO
"https://github.com/intel-innersource/applications.compilers.source.opencl-icd-loader")

set(OCL_HEADERS_TAG 9ddb236e6eb3cf844f9e2f81677e1045f9bf838e)
set(OCL_LOADER_TAG 9a3e962f16f5097d2054233ad8b6dad51b6f41b7)


FetchContent_Declare(ocl-headers
GIT_REPOSITORY    ${OCL_HEADERS_REPO}
GIT_TAG           ${OCL_HEADERS_TAG}
)

FetchContent_MakeAvailable(ocl-headers)
FetchContent_GetProperties(ocl-headers)
set(OpenCL_INCLUDE_DIR
  ${ocl-headers_SOURCE_DIR} CACHE PATH "Path to OpenCL Headers")

target_compile_definitions(Headers INTERFACE -DCL_TARGET_OPENCL_VERSION=300)
add_library(OpenCL-Headers ALIAS Headers)

set(OPENCL_ICD_LOADER_HEADERS_DIR
  ${OpenCL_INCLUDE_DIR} CACHE PATH "Path to OpenCL Headers")
  set(BUILD_SHARED_LIBS ON)
  FetchContent_Declare(ocl-icd
    GIT_REPOSITORY    ${OCL_LOADER_REPO}
    GIT_TAG           ${OCL_LOADER_TAG}
  )

FetchContent_MakeAvailable(ocl-icd)

add_library(OpenCL-ICD ALIAS OpenCL)

#     find_package(OpenCL REQUIRED)
    message(STATUS "OpenCL_LIBRARY: ${OpenCL_LIBRARY}")
    message(STATUS "OpenCL_INCLUDE_DIR: ${OpenCL_INCLUDE_DIR}")
    set(UR_OPENCL_ICD_LOADER_LIBRARY OpenCL::OpenCL)
endif()

# Suppress a compiler message about undefined CL_TARGET_OPENCL_VERSION.
# Define all symbols up to OpenCL 3.0.
target_compile_definitions(ur_adapter_opencl PRIVATE CL_TARGET_OPENCL_VERSION=300 CL_USE_DEPRECATED_OPENCL_1_2_APIS)

target_link_libraries(${TARGET_NAME} PRIVATE
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::common
        ${PROJECT_NAME}::unified_malloc_framework
        Threads::Threads
        ${UR_OPENCL_ICD_LOADER_LIBRARY}
)

target_include_directories(${TARGET_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../"
        ${OpenCL_INCLUDE_DIR}
)
