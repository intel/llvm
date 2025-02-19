# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_executable(test-loader-platforms
    platforms.cpp
)

target_link_libraries(test-loader-platforms
    PRIVATE
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::loader
    gmock
    GTest::gtest_main
    unit_tests_helpers
)

function(add_loader_platform_test name ENV)
    set(TEST_NAME loader_platform_test_${name})
    add_test(NAME ${TEST_NAME}
        COMMAND ${CMAKE_COMMAND}
        -D TEST_FILE=$<TARGET_FILE:test-loader-platforms>
        -D MODE=stdout
        -D MATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}.match
        -P ${PROJECT_SOURCE_DIR}/cmake/match.cmake
    )
    set_tests_properties(${TEST_NAME} PROPERTIES
        LABELS "loader"
        ENVIRONMENT "${ENV};UR_LOG_TEST=level:info\;output:stdout\;flush:debug"
    )
endfunction()

# Disabling the force load due to issues with the test on windows.
#add_loader_platform_test(no_platforms "UR_ADAPTERS_FORCE_LOAD=\"\"")
add_loader_platform_test(null_platform "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_mock>\"")
