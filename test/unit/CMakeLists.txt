# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_library(unit_tests_helpers INTERFACE)

target_include_directories(unit_tests_helpers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

function(add_unit_test name)
    set(TEST_TARGET_NAME test-${name})

    add_executable(${TEST_TARGET_NAME}
        ${ARGN}
    )
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        ${PROJECT_NAME}::common
        ${PROJECT_NAME}::headers
        gmock
        GTest::gtest_main
        unit_tests_helpers
    )
    add_test(NAME unit-${name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(unit-${name} PROPERTIES LABELS "unit")
endfunction()

add_subdirectory(utils)
add_subdirectory(logger)
