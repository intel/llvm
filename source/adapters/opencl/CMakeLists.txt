# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(OPENCL_DIR "${SYCL_ADAPTER_DIR}/sycl/plugins/unified_runtime/ur/adapters/opencl" CACHE PATH "OpenCL adapter directory")

set(TARGET_NAME ur_adapter_opencl)

add_ur_adapter(${TARGET_NAME}
        SHARED
        ${OPENCL_DIR}/ur_interface_loader.cpp
        ${OPENCL_DIR}/adapter.hpp
        ${OPENCL_DIR}/adapter.cpp
        ${OPENCL_DIR}/command_buffer.hpp
        ${OPENCL_DIR}/command_buffer.cpp
        ${OPENCL_DIR}/common.hpp
        ${OPENCL_DIR}/common.cpp
        ${OPENCL_DIR}/context.cpp
        ${OPENCL_DIR}/context.hpp
        ${OPENCL_DIR}/device.cpp
        ${OPENCL_DIR}/device.hpp
        ${OPENCL_DIR}/enqueue.cpp
        ${OPENCL_DIR}/event.cpp
        ${OPENCL_DIR}/image.cpp
        ${OPENCL_DIR}/kernel.cpp
        ${OPENCL_DIR}/memory.cpp
        ${OPENCL_DIR}/platform.cpp
        ${OPENCL_DIR}/platform.hpp
        ${OPENCL_DIR}/program.cpp
        ${OPENCL_DIR}/queue.cpp
        ${OPENCL_DIR}/sampler.cpp
        ${OPENCL_DIR}/usm.cpp
        ${OPENCL_DIR}/usm_p2p.cpp
        ${OPENCL_DIR}/../../ur.cpp
        ${OPENCL_DIR}/../../ur.hpp
)

set_target_properties(${TARGET_NAME} PROPERTIES
        VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
        SOVERSION "${PROJECT_VERSION_MAJOR}"
)

find_package(Threads REQUIRED)

if (NOT DEFINED OpenCL_LIBRARY OR NOT DEFINED OpenCL_INCLUDE_DIR)
    message(WARNING "OpenCL_LIBRARY and OpenCL_INCLUDE_DIR are not set. Using find_package() to find an OpenCL installation in the system.")
    find_package(OpenCL REQUIRED)
endif()

message(STATUS "OpenCL_LIBRARY: ${OpenCL_LIBRARY}")
message(STATUS "OpenCL_INCLUDE_DIR: ${OpenCL_INCLUDE_DIR}")

# Suppress a compiler message about undefined CL_TARGET_OPENCL_VERSION.
# Define all symbols up to OpenCL 3.0.
target_compile_definitions(ur_adapter_opencl PRIVATE CL_TARGET_OPENCL_VERSION=300)

# Make imported library global to use it within the project.
add_library(OpenCL-ICD SHARED IMPORTED GLOBAL)

if (WIN32)
    set_target_properties(
            OpenCL-ICD PROPERTIES
            IMPORTED_IMPLIB               ${OpenCL_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${OpenCL_INCLUDE_DIR}
    )
else()
    set_target_properties(
            OpenCL-ICD PROPERTIES
            IMPORTED_LOCATION             ${OpenCL_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${OpenCL_INCLUDE_DIR}
    )
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::common
        ${PROJECT_NAME}::unified_malloc_framework
        Threads::Threads
        OpenCL-ICD
)

target_include_directories(${TARGET_NAME} PRIVATE
        ${OPENCL_DIR}/../../../
        ${OpenCL_INCLUDE_DIR}
)
