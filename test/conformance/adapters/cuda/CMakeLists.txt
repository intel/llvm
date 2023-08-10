# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_test_with_devices_environment(adapter-cuda 
    cuda_fixtures.h
    context_tests.cpp
    cuda_urContextGetNativeHandle.cpp
    cuda_urDeviceGetNativeHandle.cpp
    cuda_urDeviceCreateWithNativeHandle.cpp
    cuda_urEventGetNativeHandle.cpp
    cuda_urEventCreateWithNativeHandle.cpp
    kernel_tests.cpp
    memory_tests.cpp
)
target_link_libraries(test-adapter-cuda PRIVATE cudadrv ur_adapter_cuda)
target_include_directories(test-adapter-cuda PRIVATE  ${CUDA_DIR} "${CUDA_DIR}/../../../" )
# TODO - remove as part of #789
target_compile_definitions(test-adapter-cuda PRIVATE UR_INCLUDE_HANDLE_IMPLEMENTATION_OVERLOADS)

set_tests_properties(adapter-cuda PROPERTIES
        LABELS "conformance:cuda"
        ENVIRONMENT "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_cuda>\""
    )
