# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

set(UR_VALIDATION_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(add_validation_test name)
    set(TEST_TARGET_NAME validation_test-${name})
    add_executable(${TEST_TARGET_NAME}
        ${ARGN})
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        ${PROJECT_NAME}::loader
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::testing
        GTest::gtest_main)
    add_test(NAME ${name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(${name} PROPERTIES LABELS "validation")
    set_property(TEST ${name} PROPERTY ENVIRONMENT
        "UR_ENABLE_VALIDATION_LAYER=1"
        "UR_ENABLE_PARAMETER_VALIDATION=1"
        "UR_ADAPTERS_FORCE_LOAD=$<TARGET_FILE:ur_adapter_null>")
endfunction()

add_validation_test(parameters parameters.cpp)
