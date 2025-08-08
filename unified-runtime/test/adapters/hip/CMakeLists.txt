# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT UR_FOUND_DPCXX)
    message(WARNING
        "UR_DPCXX is not defined, skipping adapter-specific tests for HIP")
    return()
endif()

add_ur_lit_testsuite(hip)
add_dependencies(check-unified-runtime-adapter check-unified-runtime-hip)

add_conformance_kernels_test(adapter-hip
    fixtures.h
    urContextGetNativeHandle.cpp
    urDeviceGetNativeHandle.cpp
    urEventGetNativeHandle.cpp
    test_context.cpp
    test_event.cpp
    kernel_tests.cpp
)

target_include_directories(adapter-hip-test PRIVATE
    ${PROJECT_SOURCE_DIR}/source
    ${PROJECT_SOURCE_DIR}/source/adapters/hip
)

get_target_property(HIP_COMPILE_DEFINITIONS
    ur_adapter_hip COMPILE_DEFINITIONS)
target_compile_definitions(adapter-hip-test PRIVATE
    ${HIP_COMPILE_DEFINITIONS}
)

target_link_libraries(adapter-hip-test PRIVATE rocmdrv ${PROJECT_NAME}::umf)
