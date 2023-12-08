# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_sanitizer_test name)
    set(TEST_TARGET_NAME sanitizer_test-${name})
    add_ur_executable(${TEST_TARGET_NAME}
        ${ARGN})
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        ${PROJECT_NAME}::loader
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::testing
        GTest::gtest_main)

    add_test(NAME test_${name}
        COMMAND ${CMAKE_COMMAND}
        -D MODE=stderr
        -D TEST_FILE=$<TARGET_FILE:${TEST_TARGET_NAME}>
        -D MATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.out.match
        -P ${PROJECT_SOURCE_DIR}/cmake/match.cmake
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    set_tests_properties(test_${name} PROPERTIES
        LABELS ${name}
    )

    string(TOUPPER ${name} upper_name)
    set_property(TEST test_${name} PROPERTY ENVIRONMENT
    "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_null>\"")
endfunction()

add_sanitizer_test(asan test_asan.cpp)
