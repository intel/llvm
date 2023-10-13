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

if (NOT DEFINED OpenCL_LIBRARY OR NOT DEFINED OpenCL_INCLUDE_DIR)
    message(WARNING "OpenCL_LIBRARY and OpenCL_INCLUDE_DIR are not set. Using find_package() to find an OpenCL installation in the system.")
    find_package(OpenCL REQUIRED)
endif()

message(STATUS "OpenCL_LIBRARY: ${OpenCL_LIBRARY}")
message(STATUS "OpenCL_INCLUDE_DIR: ${OpenCL_INCLUDE_DIR}")

# Suppress a compiler message about undefined CL_TARGET_OPENCL_VERSION.
# Define all symbols up to OpenCL 3.0.
target_compile_definitions(ur_adapter_opencl PRIVATE CL_TARGET_OPENCL_VERSION=300 CL_USE_DEPRECATED_OPENCL_1_2_APIS)

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
        "${CMAKE_CURRENT_SOURCE_DIR}/../../"
        ${OpenCL_INCLUDE_DIR}
)
