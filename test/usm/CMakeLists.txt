# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(UR_USM_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(add_usm_test name)
    set(TEST_TARGET_NAME usm_test-${name})
    add_ur_executable(${TEST_TARGET_NAME}
        ${UR_USM_TEST_DIR}/../conformance/source/environment.cpp
        ${UR_USM_TEST_DIR}/../conformance/source/main.cpp
        ${ARGN})
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        ${PROJECT_NAME}::common
        ${PROJECT_NAME}::loader
        ur_testing
        GTest::gtest_main)
    add_test(NAME usm-${name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(usm-${name} PROPERTIES
        LABELS "usm"
        ENVIRONMENT "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_null>\"")
    target_compile_definitions("usm_test-${name}" PRIVATE DEVICES_ENVIRONMENT)
endfunction()

add_usm_test(usmPoolManager usmPoolManager.cpp)
