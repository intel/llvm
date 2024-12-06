# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(UR_CONFORMANCE_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(UR_CONFORMANCE_DEVICE_BINARIES_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/device_binaries" CACHE INTERNAL
    "Internal cache variable for device binaries directory")

function(add_test_adapter name adapter backend)
    set(TEST_TARGET_NAME test-${name})
    set(TEST_NAME ${name}-${adapter})

    set(TEST_COMMAND
        "${PROJECT_BINARY_DIR}/bin/${TEST_TARGET_NAME} --backend=${backend} --devices_count=${UR_TEST_DEVICES_COUNT} --platforms_count=${UR_TEST_PLATFORMS_COUNT}"
    )
    set(MATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${name}_${adapter}.match")

    function(do_add_test tname env)
        if(${UR_CONFORMANCE_ENABLE_MATCH_FILES} AND EXISTS ${MATCH_FILE})
            add_test(NAME ${tname}
                COMMAND ${Python3_EXECUTABLE} ${UR_CONFORMANCE_TEST_DIR}/cts_exe.py
                    --failslist ${MATCH_FILE}
                    --test_command ${PROJECT_BINARY_DIR}/bin/${TEST_TARGET_NAME}
                    --
                    --backend=${backend}
                    --devices_count=${UR_TEST_DEVICES_COUNT}
                    --platforms_count=${UR_TEST_PLATFORMS_COUNT}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
        else()
            separate_arguments(TEST_COMMAND)
            add_test(NAME ${tname}
                COMMAND ${TEST_COMMAND}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
        endif()

        if(UR_CONFORMANCE_ENABLE_MATCH_FILES)
            list(APPEND env GTEST_COLOR=yes)
        endif()
        set_tests_properties(${tname} PROPERTIES
            ENVIRONMENT "${env}"
            LABELS "conformance;${adapter}")
    endfunction()

    do_add_test(${TEST_NAME} UR_ADAPTERS_FORCE_LOAD="$<TARGET_FILE:ur_${adapter}>")
    if(UR_CONFORMANCE_TEST_LOADER)
        do_add_test(${TEST_NAME}-loader "")
    endif()
endfunction()

function(add_conformance_test name)
    set(TEST_TARGET_NAME test-${name})
    add_ur_executable(${TEST_TARGET_NAME}
        ${ARGN}
        ${UR_CONFORMANCE_TEST_DIR}/source/environment.cpp
        ${UR_CONFORMANCE_TEST_DIR}/source/main.cpp)
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        ${PROJECT_NAME}::loader
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::testing
        ${PROJECT_NAME}::common
        GTest::gtest_main
        unit_tests_helpers)

    if(UR_BUILD_ADAPTER_CUDA OR UR_BUILD_ADAPTER_ALL)
        add_test_adapter(${name} adapter_cuda CUDA)
    endif()
    if(UR_BUILD_ADAPTER_HIP OR UR_BUILD_ADAPTER_ALL)
        add_test_adapter(${name} adapter_hip HIP)
    endif()
    if(UR_BUILD_ADAPTER_L0 OR UR_BUILD_ADAPTER_ALL)
        add_test_adapter(${name} adapter_level_zero LEVEL_ZERO)
    endif()
    if(UR_BUILD_ADAPTER_L0_V2)
        add_test_adapter(${name} adapter_level_zero_v2 LEVEL_ZERO)
    endif()
    if(UR_BUILD_ADAPTER_OPENCL OR UR_BUILD_ADAPTER_ALL)
        add_test_adapter(${name} adapter_opencl OPENCL)
    endif()
    if(UR_BUILD_ADAPTER_NATIVE_CPU OR UR_BUILD_ADAPTER_ALL)
        add_test_adapter(${name} adapter_native_cpu NATIVE_CPU)
    endif()
endfunction()

function(add_conformance_test_with_kernels_environment name)
    add_conformance_test(${name} ${ARGN})
    target_compile_definitions("test-${name}" PRIVATE KERNELS_ENVIRONMENT PRIVATE
      KERNELS_DEFAULT_DIR="${UR_CONFORMANCE_DEVICE_BINARIES_DIR}")
    target_include_directories("test-${name}" PRIVATE ${UR_CONFORMANCE_DEVICE_BINARIES_DIR})
    add_dependencies("test-${name}" generate_device_binaries)
endfunction()

function(add_conformance_test_with_devices_environment name)
    add_conformance_test(${name} ${ARGN})
    target_compile_definitions("test-${name}" PRIVATE DEVICES_ENVIRONMENT)
endfunction()

function(add_conformance_test_with_platform_environment name)
    add_conformance_test(${name} ${ARGN})
    target_compile_definitions("test-${name}" PRIVATE PLATFORM_ENVIRONMENT)
endfunction()

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
    "exp_usm_p2p"
    "exp_launch_properties"
    "memory-migrate"
    "usm"
)

if(UR_DPCXX)
    add_custom_target(generate_device_binaries)

    set(UR_CONFORMANCE_DEVICE_BINARIES_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/device_binaries" CACHE INTERNAL UR_CONFORMANCE_DEVICE_BINARIES_DIR)
    file(MAKE_DIRECTORY ${UR_CONFORMANCE_DEVICE_BINARIES_DIR})

    if("${UR_CONFORMANCE_TARGET_TRIPLES}" STREQUAL "")
        if(UR_BUILD_ADAPTER_L0 OR UR_BUILD_ADAPTER_L0_V2 OR UR_BUILD_ADAPTER_OPENCL OR UR_BUILD_ADAPTER_ALL)
            list(APPEND TARGET_TRIPLES "spir64")
        endif()
        if(UR_BUILD_ADAPTER_CUDA OR UR_BUILD_ADAPTER_ALL)
            list(APPEND TARGET_TRIPLES "nvptx64-nvidia-cuda")
        endif()
        if(UR_BUILD_ADAPTER_HIP OR UR_BUILD_ADAPTER_ALL)
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
