# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT UR_FOUND_DPCXX)
    message(WARNING
        "UR_DPCXX is not defined, skipping adapter-specific tests for Cuda")
    return()
endif()

add_ur_lit_testsuite(cuda)
add_dependencies(check-unified-runtime-adapter check-unified-runtime-cuda)

add_conformance_devices_test(adapter-cuda
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
)

target_include_directories(adapter-cuda-test PRIVATE
    ${PROJECT_SOURCE_DIR}/source
    ${PROJECT_SOURCE_DIR}/source/adapters/cuda
)

find_package(CUDAToolkit 10.1 REQUIRED)

target_link_libraries(adapter-cuda-test PRIVATE cudadrv CUDA::nvml ${PROJECT_NAME}::umf)
