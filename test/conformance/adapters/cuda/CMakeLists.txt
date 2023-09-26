# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(USE_LOCAL_TEST_CONFIGURATION ON)
add_conformance_test_with_devices_environment(adapter-cuda
    fixtures.h
    context_tests.cpp
    urContextGetNativeHandle.cpp
    urDeviceGetNativeHandle.cpp
    urDeviceCreateWithNativeHandle.cpp
    urEventGetNativeHandle.cpp
    urEventCreateWithNativeHandle.cpp
    kernel_tests.cpp
    memory_tests.cpp
)
target_link_libraries(test-adapter-cuda PRIVATE cudadrv ur_adapter_cuda)
target_include_directories(test-adapter-cuda PRIVATE  ${CUDA_DIR} "${CUDA_DIR}/../../" )

set_tests_properties(adapter-cuda PROPERTIES
        LABELS "conformance:cuda"
        ENVIRONMENT "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_cuda>\""
    )
