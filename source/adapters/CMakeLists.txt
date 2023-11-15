# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_ur_adapter name)
    add_ur_library(${name} ${ARGN})
    if(MSVC)
        set(TARGET_LIBNAME ${name})
        string(TOUPPER ${TARGET_LIBNAME} TARGET_LIBNAME)

        set(ADAPTER_VERSION_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/${name}.def)

        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../adapter.def.in ${ADAPTER_VERSION_SCRIPT} @ONLY)
        set_target_properties(${name} PROPERTIES
            LINK_FLAGS "/DEF:${ADAPTER_VERSION_SCRIPT}"
        )
    elseif(APPLE)
        target_compile_options(${name} PRIVATE "-fvisibility=hidden")
    else()
        set(TARGET_LIBNAME lib${name}_${PROJECT_VERSION_MAJOR}.0)
        string(TOUPPER ${TARGET_LIBNAME} TARGET_LIBNAME)

        set(ADAPTER_VERSION_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/${name}.map)

        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../adapter.map.in ${ADAPTER_VERSION_SCRIPT} @ONLY)
        target_link_options(${name} PRIVATE "-Wl,--version-script=${ADAPTER_VERSION_SCRIPT}")
    endif()
endfunction()

add_subdirectory(null)

set(INTEL_LLVM_TAG nightly-2023-09-20)

if(UR_BUILD_ADAPTER_L0)
    add_subdirectory(level_zero)
endif()

if(UR_BUILD_ADAPTER_CUDA)
    add_subdirectory(cuda)
endif()

if(UR_BUILD_ADAPTER_HIP)
    add_subdirectory(hip)
endif()

if(UR_BUILD_ADAPTER_OPENCL)
    add_subdirectory(opencl)
endif()
if(UR_BUILD_ADAPTER_NATIVE_CPU)
    add_subdirectory(native_cpu)
endif()
