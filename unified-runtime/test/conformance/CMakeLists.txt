# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_lit_testsuite(conformance DEPENDS urinfo)

set(UR_CONFORMANCE_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL
    "Internal cache variable for conformance test root")

set(UR_CONFORMANCE_DEVICE_BINARIES_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/device_binaries" CACHE INTERNAL
    "Internal cache variable for device binaries directory")

function(add_conformance_test name)
    add_gtest_test(${name}
        ${ARGN}
        ${UR_CONFORMANCE_TEST_DIR}/source/environment.cpp
        ${UR_CONFORMANCE_TEST_DIR}/source/main.cpp
    )
endfunction()

function(add_conformance_kernels_test name)
    add_conformance_test(${name} ${ARGN})
    target_compile_definitions(${name}-test PRIVATE KERNELS_ENVIRONMENT PRIVATE
      KERNELS_DEFAULT_DIR="${UR_CONFORMANCE_DEVICE_BINARIES_DIR}")
    target_include_directories(${name}-test PRIVATE ${UR_CONFORMANCE_DEVICE_BINARIES_DIR})
    add_dependencies(${name}-test generate_device_binaries kernel_names_header)
endfunction()

function(add_conformance_devices_test name)
    add_conformance_test(${name} ${ARGN})
    target_compile_definitions(${name}-test PRIVATE DEVICES_ENVIRONMENT)
endfunction()

function(add_conformance_platform_test name)
    add_conformance_test(${name} ${ARGN})
    target_compile_definitions(${name}-test PRIVATE PLATFORM_ENVIRONMENT)
endfunction()

foreach(adapter ${UR_ADAPTERS_LIST})
    if(NOT "${adapter}" STREQUAL "mock")
        if(UR_STANDALONE_BUILD)
            add_custom_target(check-unified-runtime-conformance-${adapter}
                COMMAND "${URLIT_LIT_BINARY}" "${CMAKE_CURRENT_BINARY_DIR}"
                    -v -Dselector=${adapter}:*
                DEPENDS deps_check-unified-runtime-conformance
            )
        else()
            add_lit_testsuite(check-unified-runtime-conformance-${adapter}
                "Running Unified Runtime ${adapter} conformance tests"
                ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS FileCheck deps_check-unified-runtime-conformance
                PARAMS selector=${adapter}:*
                EXCLUDE_FROM_CHECK_ALL
            )
        endif()
    endif()
endforeach()

add_subdirectory(testing)

add_subdirectory(adapter)
add_subdirectory(platform)
add_subdirectory(device)
add_subdirectory(context)
add_subdirectory(memory)
add_subdirectory(event)
add_subdirectory(queue)
add_subdirectory(sampler)
add_subdirectory(virtual_memory)

set(TEST_SUBDIRECTORIES_DPCXX
    "device_code"
    "kernel"
    "program"
    "enqueue"
    "integration"
    "exp_command_buffer"
    "exp_enqueue_native"
    "exp_kernel_launch_with_args"
    "exp_usm_p2p"
    "memory-migrate"
    "usm"
)

if(UR_FOUND_DPCXX)
    add_custom_target(generate_device_binaries)

    set(UR_CONFORMANCE_DEVICE_BINARIES_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/device_binaries" CACHE INTERNAL UR_CONFORMANCE_DEVICE_BINARIES_DIR)
    file(MAKE_DIRECTORY ${UR_CONFORMANCE_DEVICE_BINARIES_DIR})

    if("${UR_CONFORMANCE_TARGET_TRIPLES}" STREQUAL "")
        if(UR_BUILD_ADAPTER_L0 OR UR_BUILD_ADAPTER_OFFLOAD OR UR_BUILD_ADAPTER_L0_V2 OR UR_BUILD_ADAPTER_OPENCL OR UR_BUILD_ADAPTER_ALL)
            list(APPEND TARGET_TRIPLES "spir64")
        endif()
        if(UR_BUILD_ADAPTER_CUDA OR UR_BUILD_ADAPTER_OFFLOAD OR UR_BUILD_ADAPTER_ALL)
            list(APPEND TARGET_TRIPLES "nvptx64-nvidia-cuda")
        endif()
        if(UR_BUILD_ADAPTER_HIP OR UR_BUILD_ADAPTER_OFFLOAD OR UR_BUILD_ADAPTER_ALL)
            list(APPEND TARGET_TRIPLES "amdgcn-amd-amdhsa")
        endif()
    else()
        string(REPLACE "," ";" TARGET_TRIPLES ${UR_CONFORMANCE_TARGET_TRIPLES})
    endif()

    foreach(dir ${TEST_SUBDIRECTORIES_DPCXX})
        add_subdirectory(${dir})
    endforeach()
else()
    set(DISABLED_TESTS "")
    foreach(dir ${TEST_SUBDIRECTORIES_DPCXX})
        if(NOT dir STREQUAL "device_code")
            list(APPEND DISABLED_TESTS "test-${dir}")
        endif()
    endforeach()

    string(REPLACE ";" ", " DISABLED_TESTS_STR "${DISABLED_TESTS}")

    message(WARNING
        "UR_DPCXX is not defined, the following conformance test executables \
        are disabled: ${DISABLED_TESTS_STR}")
endif()
