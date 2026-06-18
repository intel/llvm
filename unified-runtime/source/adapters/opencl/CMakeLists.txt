# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(OPENCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "OpenCL adapter directory")

include(FetchOpenCL)

find_package(Threads REQUIRED)

set(TARGET_NAME ur_adapter_opencl)

# Determine adapter library type (SHARED or STATIC)
set(ADAPTER_LIB_TYPE SHARED)
if(UR_STATIC_ADAPTER_OPENCL)
    set(ADAPTER_LIB_TYPE STATIC)
endif()

add_ur_adapter(${TARGET_NAME} ${ADAPTER_LIB_TYPE}
    ${CMAKE_CURRENT_SOURCE_DIR}/ur_interface_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/async_alloc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_buffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/context.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue_native.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory_export.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sampler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/usm.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/usm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/usm_p2p.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/virtual_mem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.hpp
)
install_ur_library(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
)

# https://github.com/intel/llvm/issues/19648
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND LLVM_ENABLE_PEDANTIC)
    target_compile_options(ur_adapter_opencl PRIVATE -std=gnu++17)
endif()

target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../"
)

if(UR_STATIC_ADAPTER_OPENCL)
    # When building as static library, add dynamic loading source files
    # and don't link against OpenCL library
    target_sources(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ocl_dynamic_lib.cpp
    )
    target_compile_definitions(${TARGET_NAME} PRIVATE
        UR_STATIC_ADAPTER_OPENCL
        # Neutralize dllexport/dllimport annotations on MSVC: the generated .cpp
        # files use UR_APIEXPORT/UR_APICALL but a static library must not have
        # __declspec(dllimport) on definitions (causes C2375) and does not need
        # __declspec(dllexport) either.
        UR_APIEXPORT=
        UR_APICALL=
        UR_DLLEXPORT=
    )
    target_link_libraries(${TARGET_NAME} PRIVATE
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::common
        ${PROJECT_NAME}::umf
        Threads::Threads
    )
    # OpenCL-Headers is header-only; use its include path directly to avoid
    # exporting a target whose INTERFACE_INCLUDE_DIRECTORIES points into the
    # build tree, which CMake rejects in install(EXPORT ...).
    target_include_directories(${TARGET_NAME} PRIVATE
        $<TARGET_PROPERTY:OpenCL-Headers,INTERFACE_INCLUDE_DIRECTORIES>)
    # Add dl library for dynamic loading on Unix systems
    if(UNIX)
        target_link_libraries(${TARGET_NAME} PRIVATE ${CMAKE_DL_LIBS})
    endif()
else()
    # Standard shared library build - link against OpenCL library
    target_link_libraries(${TARGET_NAME} PRIVATE
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::common
        ${PROJECT_NAME}::umf
        Threads::Threads
        OpenCL-Headers
        ${OpenCL_LIBRARY}
    )
endif()
