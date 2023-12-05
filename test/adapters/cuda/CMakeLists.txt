# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_adapter_test(cuda
    FIXTURE DEVICES
    SOURCES
        fixtures.h
        context_tests.cpp
        urContextGetNativeHandle.cpp
        urDeviceGetNativeHandle.cpp
        urDeviceCreateWithNativeHandle.cpp
        urEventGetNativeHandle.cpp
        urEventCreateWithNativeHandle.cpp
        kernel_tests.cpp
        memory_tests.cpp
    ENVIRONMENT
        "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_cuda>\""
)

target_include_directories(test-adapter-cuda PRIVATE
    ${PROJECT_SOURCE_DIR}/source
    ${PROJECT_SOURCE_DIR}/source/adapters/cuda
)

target_link_libraries(test-adapter-cuda PRIVATE cudadrv)
