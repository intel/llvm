# Copyright (C) 2023 Intel Corporation
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

if(UR_BUILD_ADAPTER_CUDA OR UR_BUILD_ADAPTER_ALL)
    add_subdirectory(cuda)
endif()

if(UR_BUILD_ADAPTER_HIP OR UR_BUILD_ADAPTER_ALL)
    add_subdirectory(hip)
endif()
