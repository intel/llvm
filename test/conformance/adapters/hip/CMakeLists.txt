# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(USE_LOCAL_TEST_CONFIGURATION ON)
add_conformance_test_with_devices_environment(adapter-hip 
    fixtures.h
    urContextGetNativeHandle.cpp
    urDeviceGetNativeHandle.cpp
    urEventGetNativeHandle.cpp
    test_context.cpp
)
target_link_libraries(test-adapter-hip PRIVATE rocmdrv)
target_include_directories(test-adapter-hip PRIVATE ${HIP_DIR} "${HIP_DIR}/../../")

if(UR_HIP_PLATFORM STREQUAL "AMD")
    target_compile_definitions(test-adapter-hip PRIVATE __HIP_PLATFORM_AMD__)
else()
    target_compile_definitions(test-adapter-hip PRIVATE __HIP_PLATFORM_NVIDIA__)
endif()

set_tests_properties(adapter-hip PROPERTIES
    LABELS "conformance:hip"
    ENVIRONMENT "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_hip>\""
)
