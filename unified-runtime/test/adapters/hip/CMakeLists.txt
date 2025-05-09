# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT UR_FOUND_DPCXX)
    message(WARNING
        "UR_DPCXX is not defined, skipping adapter-specific tests for HIP")
    return()
endif()

add_adapter_test(hip
    FIXTURE KERNELS
    SOURCES
        fixtures.h
        urContextGetNativeHandle.cpp
        urDeviceGetNativeHandle.cpp
        urEventGetNativeHandle.cpp
        test_context.cpp
        test_event.cpp
        kernel_tests.cpp
    ENVIRONMENT
        "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_hip>\""
)

target_include_directories(test-adapter-hip PRIVATE
    ${PROJECT_SOURCE_DIR}/source
    ${PROJECT_SOURCE_DIR}/source/adapters/hip
)

get_target_property(HIP_COMPILE_DEFINITIONS
    ur_adapter_hip COMPILE_DEFINITIONS)
target_compile_definitions(test-adapter-hip PRIVATE
    ${HIP_COMPILE_DEFINITIONS}
)

target_link_libraries(test-adapter-hip PRIVATE rocmdrv ${PROJECT_NAME}::umf)
