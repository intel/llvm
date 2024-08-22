# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_adapter_test name)
    cmake_parse_arguments(args
        ""                      # options
        "FIXTURE"               # one value keywords
        "SOURCES;ENVIRONMENT"   # multi value keywords
        ${ARGN})

    set(target test-adapter-${name})
    add_ur_executable(${target} ${args_SOURCES}
        ${PROJECT_SOURCE_DIR}/test/conformance/source/environment.cpp
        ${PROJECT_SOURCE_DIR}/test/conformance/source/main.cpp
    )

    set(fixtures "PLATFORM;DEVICES;KERNELS")
    if(NOT args_FIXTURE IN_LIST fixtures)
        message(FATAL_ERROR
            "FIXTURE must be one of: ${fixtures}. Found: ${args_FIXTURE}")
    endif()

    target_compile_definitions(${target} PRIVATE
        ${args_FIXTURE}_ENVIRONMENT)

    if(${args_FIXTURE} STREQUAL "KERNELS")
        target_compile_definitions(${target} PRIVATE KERNELS_ENVIRONMENT PRIVATE
            KERNELS_DEFAULT_DIR="${UR_CONFORMANCE_DEVICE_BINARIES_DIR}")
        target_include_directories(${target}
            PRIVATE ${UR_CONFORMANCE_DEVICE_BINARIES_DIR})
        add_dependencies(${target} generate_device_binaries kernel_names_header)
    endif()

    target_link_libraries(${target} PRIVATE
        ${PROJECT_NAME}::loader
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::testing
        ${PROJECT_NAME}::common
        GTest::gtest)

    add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}>
        --devices_count=${UR_TEST_DEVICES_COUNT}
        --platforms_count=${UR_TEST_DEVICES_COUNT})
    set_tests_properties(${target} PROPERTIES
        LABELS "adapter-specific;${name}"
        ENVIRONMENT "${args_ENVIRONMENT}")
endfunction()

find_program(VALGRIND valgrind)

function(add_adapter_memcheck_test name)
    cmake_parse_arguments(args
        ""                      # options
        ""                      # one value keywords
        "ENVIRONMENT"           # multi value keywords
        ${ARGN})
    if(VALGRIND)
        set(target test-adapter-${name})
        set(test_name ${target}-memcheck)

        add_test(NAME ${test_name}
                COMMAND ${CMAKE_COMMAND}
                -D TEST_FILE=valgrind
                -D TEST_ARGS="--tool=memcheck --leak-check=full $<TARGET_FILE:${target}> --devices_count=${UR_TEST_DEVICES_COUNT} --platforms_count=${UR_TEST_DEVICES_COUNT}"
                -D MODE=stderr
                -D MATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}_memcheck.match
                -P ${PROJECT_SOURCE_DIR}/cmake/match.cmake
                DEPENDS ${TEST_TARGET_NAME}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )

        set_tests_properties(${test_name} PROPERTIES
            LABELS "adapter-specific;${name}"
            ENVIRONMENT "${args_ENVIRONMENT}")
    endif()
endfunction()

if(UR_BUILD_ADAPTER_CUDA OR UR_BUILD_ADAPTER_ALL)
    add_subdirectory(cuda)
endif()

if(UR_BUILD_ADAPTER_HIP OR UR_BUILD_ADAPTER_ALL)
    add_subdirectory(hip)
endif()

if(UR_BUILD_ADAPTER_L0 OR UR_BUILD_ADAPTER_L0_V2 OR UR_BUILD_ADAPTER_ALL)
    add_subdirectory(level_zero)
endif()
