# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_lit_testsuite(level_zero)
add_dependencies(check-unified-runtime-adapter check-unified-runtime-level_zero)

# Level Zero tests use the ShTest runner rather than the gtest runner since many tests
# have requirements and special options

function(add_l0_loader_kernels_test name)
    add_conformance_kernels_test(${name} ${ARGN})
    target_link_libraries(${name}-test PRIVATE
        LevelZeroLoader
        LevelZeroLoader-Headers
        ComputeRuntimeLevelZero-Headers
    )

    target_include_directories(${name}-test PRIVATE
        LevelZeroLoader-Headers
    )
endfunction()

function(add_l0_loader_devices_test name)
    add_conformance_devices_test(${name} ${ARGN})
    target_link_libraries(${name}-test PRIVATE
        LevelZeroLoader
        LevelZeroLoader-Headers
        ComputeRuntimeLevelZero-Headers
    )

    target_include_directories(${name}-test PRIVATE
        LevelZeroLoader-Headers
    )
endfunction()

if(NOT UR_FOUND_DPCXX)
    # Tests that require kernels can't be used if we aren't generating
    # device binaries
    message(WARNING
        "UR_DPCXX is not defined, skipping some adapter tests for ${adapter}")
else()
    add_conformance_kernels_test(link urProgramLink.cpp)
    add_l0_loader_kernels_test(kernel_create urKernelCreateWithNativeHandle.cpp)
    add_l0_loader_kernels_test(event_create urEventCreateWithNativeHandle.cpp)
    add_conformance_kernels_test(enqueue_alloc enqueue_alloc.cpp)
endif()

add_testing_binary(confirm_version confirm_version.cpp)

add_l0_loader_devices_test(ze_calls event_cache_tests.cpp)
add_l0_loader_devices_test(multi_queue multi_device_event_cache_tests.cpp)

add_conformance_devices_test(ipc ipc.cpp)
target_link_libraries(ipc-test PRIVATE ur_umf)

add_conformance_devices_test(mem_buffer_map urEnqueueMemBufferMapHostPtr.cpp)
add_conformance_devices_test(device_select_binary urDeviceSelectBinary.cpp)

add_l0_loader_devices_test(mem_buffer_create_with_native_handle urMemBufferCreateWithNativeHandleShared.cpp)

if(UR_BUILD_ADAPTER_L0_V2)
    add_subdirectory(v2)
endif()
