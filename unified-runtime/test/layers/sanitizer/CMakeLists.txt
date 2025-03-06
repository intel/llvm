# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(UR_SANITIZER_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SAN_TEST_PREFIX sanitizer_test)

function(add_sanitizer_test_executable name)
    add_ur_executable(${SAN_TEST_PREFIX}-${name}
        ${ARGN})
    target_link_libraries(${SAN_TEST_PREFIX}-${name}
        PRIVATE
        ${PROJECT_NAME}::loader
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::testing
        ${PROJECT_NAME}::mock
        GTest::gtest_main)
endfunction()

function(set_sanitizer_test_properties name)
    set_tests_properties(${name} PROPERTIES LABELS "sanitizer")
    set_property(TEST ${name} PROPERTY ENVIRONMENT
        "UR_LOG_SANITIZER=level:debug\;flush:debug\;output:stdout")
    target_include_directories(${SAN_TEST_PREFIX}-${name} PRIVATE
        ${PROJECT_SOURCE_DIR}/source
        ${PROJECT_SOURCE_DIR}/source/common
        ${PROJECT_SOURCE_DIR}/source/loader/layers/sanitizer/sanitizer_common
    )
endfunction()

function(add_sanitizer_test name)
    add_sanitizer_test_executable(${name} ${ARGN})

    add_test(NAME ${name}
        COMMAND ${SAN_TEST_PREFIX}-${name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    set_sanitizer_test_properties(${name})
endfunction()

function(add_test_source name source)
    target_sources(${SAN_TEST_PREFIX}-${name} PRIVATE ${source})
endfunction()

add_sanitizer_test(asan asan.cpp)

add_sanitizer_test(sanitizer_options sanitizer_options.cpp)
# add the source here in order to test for unexported functions
add_test_source(sanitizer_options 
    ${PROJECT_SOURCE_DIR}/source/loader/layers/sanitizer/sanitizer_common/sanitizer_options.cpp
)
