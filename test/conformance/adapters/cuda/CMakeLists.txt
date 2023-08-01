# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_test_with_devices_environment(adapter-cuda 
    cuda_fixtures.h
    cuda_urContextGetNativeHandle.cpp
    cuda_urDeviceGetNativeHandle.cpp
    cuda_urDeviceCreateWithNativeHandle.cpp
    cuda_urEventGetNativeHandle.cpp
    cuda_urEventCreateWithNativeHandle.cpp
)
target_link_libraries(test-adapter-cuda PRIVATE cudadrv)

set_tests_properties(adapter-cuda PROPERTIES
        LABELS "conformance:cuda"
        ENVIRONMENT "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_cuda>\""
    )
