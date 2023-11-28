# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(UR_VALIDATION_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(VAL_TEST_PREFIX validation_test)

function(add_validation_test_executable name)
    add_ur_executable(${VAL_TEST_PREFIX}-${name}
        ${ARGN})
    target_link_libraries(${VAL_TEST_PREFIX}-${name}
        PRIVATE
        ${PROJECT_NAME}::loader
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::testing
        GTest::gtest_main)
endfunction()

function(set_validation_test_properties name)
    set_tests_properties(${name} PROPERTIES LABELS "validation")
    set_property(TEST ${name} PROPERTY ENVIRONMENT
        "UR_ENABLE_LAYERS=UR_LAYER_FULL_VALIDATION"
        "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_null>\""
        "UR_LOG_VALIDATION=level:debug\;flush:debug\;output:stdout")
endfunction()

function(add_validation_test name)
    add_validation_test_executable(${name} ${ARGN})

    add_test(NAME ${name}
        COMMAND ${VAL_TEST_PREFIX}-${name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    set_validation_test_properties(${name})
endfunction()

function(add_validation_match_test name)
    add_validation_test_executable(${name} ${ARGN})

    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND}
        -D MODE=stdout
        -D TEST_FILE=$<TARGET_FILE:${VAL_TEST_PREFIX}-${name}>
        -D MATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}.out.match
        -P ${PROJECT_SOURCE_DIR}/cmake/match.cmake
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    set_validation_test_properties(${name})
endfunction()

add_validation_test(parameters parameters.cpp)
add_validation_match_test(leaks leaks.out.match leaks.cpp)
add_validation_match_test(leaks_mt leaks_mt.out.match leaks_mt.cpp)
add_validation_match_test(lifetime lifetime.out.match lifetime.cpp)
