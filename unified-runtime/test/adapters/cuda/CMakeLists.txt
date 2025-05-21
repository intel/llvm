# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT UR_FOUND_DPCXX)
    message(WARNING
        "UR_DPCXX is not defined, skipping adapter-specific tests for Cuda")
    return()
endif()

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
        urQueueGetNativeHandle.cpp
        kernel_tests.cpp
        memory_tests.cpp
        event_tests.cpp
        #FIXME: make this cleaner
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../source/adapters/cuda/queue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../source/adapters/cuda/common.cpp
    ENVIRONMENT
        "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_cuda>\""
)

target_include_directories(test-adapter-cuda PRIVATE
    ${PROJECT_SOURCE_DIR}/source
    ${PROJECT_SOURCE_DIR}/source/adapters/cuda
)

find_package(CUDAToolkit 10.1 REQUIRED)

target_link_libraries(test-adapter-cuda PRIVATE cudadrv CUDA::nvml ${PROJECT_NAME}::umf)
