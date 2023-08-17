# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(UR_CONFORMANCE_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(add_test_adapter name adapter)
    set(TEST_TARGET_NAME test-${name})
    set(TEST_NAME ${name}-${adapter})

    add_test(NAME ${TEST_NAME}
        COMMAND ${CMAKE_COMMAND}
        -D TEST_FILE=${Python3_EXECUTABLE}
        -D TEST_ARGS="${UR_CONFORMANCE_TEST_DIR}/cts_exe.py --test_command ${CMAKE_BINARY_DIR}/bin/${TEST_TARGET_NAME}"
        -D MODE=stdout
        -D MATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}_${adapter}.match
        -P ${PROJECT_SOURCE_DIR}/cmake/match.cmake
        DEPENDS ${TEST_TARGET_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    set_tests_properties(${TEST_NAME} PROPERTIES
        ENVIRONMENT "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_${adapter}>\""
        LABELS "conformance;${adapter}")
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
        GTest::gtest_main)

    if(USE_LOCAL_TEST_CONFIGURATION)
	    add_test(NAME ${name}
            COMMAND ${TEST_TARGET_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    else()
        if(UR_BUILD_ADAPTER_CUDA)
            add_test_adapter(${name} adapter_cuda)
        endif()
        if(UR_BUILD_ADAPTER_HIP)
            add_test_adapter(${name} adapter_hip)
        endif()
        if(UR_BUILD_ADAPTER_L0)
            add_test_adapter(${name} adapter_level_zero)
        endif()

        if(NOT (UR_BUILD_ADAPTER_CUDA
            OR UR_BUILD_ADAPTER_HIP OR UR_BUILD_ADAPTER_L0))
            add_test_adapter(${name} adapter_null)
        endif()
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
add_subdirectory(adapters)

add_subdirectory(runtime)
add_subdirectory(platform)
add_subdirectory(device)
add_subdirectory(context)
add_subdirectory(memory)
add_subdirectory(usm)
add_subdirectory(event)
add_subdirectory(queue)
add_subdirectory(sampler)
add_subdirectory(virtual_memory)

if(DEFINED UR_DPCXX)
    add_custom_target(generate_device_binaries)

    set(UR_CONFORMANCE_DEVICE_BINARIES_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/device_binaries" CACHE INTERNAL UR_CONFORMANCE_DEVICE_BINARIES_DIR)
    file(MAKE_DIRECTORY ${UR_CONFORMANCE_DEVICE_BINARIES_DIR})

    if(NOT "${UR_CONFORMANCE_TARGET_TRIPLES}" STREQUAL "")
        string(REPLACE "," ";" TARGET_TRIPLES ${UR_CONFORMANCE_TARGET_TRIPLES})
    else()
        message(WARNING "UR_CONFORMANCE_TARGET_TRIPLES wasn't set, defaulting to only generate spir64 device binaries")
        list(APPEND TARGET_TRIPLES "spir64")
    endif()

    add_subdirectory(device_code)
    add_subdirectory(kernel)
    add_subdirectory(program)
    add_subdirectory(enqueue)
endif()
