# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_test_with_devices_environment(adapter-hip 
    hip_fixtures.h
    hip_urContextGetNativeHandle.cpp
    hip_urDeviceGetNativeHandle.cpp
    hip_urEventGetNativeHandle.cpp
)
target_link_libraries(test-adapter-hip PRIVATE rocmdrv)

if(UR_HIP_PLATFORM STREQUAL "AMD")
    target_compile_definitions(test-adapter-hip PRIVATE __HIP_PLATFORM_AMD__)
else()
    target_compile_definitions(test-adapter-hip PRIVATE __HIP_PLATFORM_NVIDIA__)
endif()

set_tests_properties(adapter-hip PROPERTIES
    LABELS "conformance:hip"
    ENVIRONMENT "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_hip>\""
)
