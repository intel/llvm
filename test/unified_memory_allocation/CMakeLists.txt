# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: MIT

set(UR_UMA_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(uma_test_helpers STATIC
    ${UR_UMA_TEST_DIR}/common/pool.c
    ${UR_UMA_TEST_DIR}/common/provider.c
)

target_link_libraries(uma_test_helpers PRIVATE ${PROJECT_NAME}::unified_memory_allocation)

function(add_uma_test name)
    set(TEST_TARGET_NAME uma_test-${name})
    add_executable(${TEST_TARGET_NAME}
        ${ARGN})
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        ${PROJECT_NAME}::unified_memory_allocation
        uma_test_helpers
        GTest::gtest_main)
    target_include_directories(${TEST_TARGET_NAME} PRIVATE
        ${UR_UMA_TEST_DIR}/common)
    add_test(NAME  uma-${name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

add_uma_test(memoryProvider memoryProvider.cpp)
add_uma_test(memoryPool memoryPool.cpp)
add_uma_test(base base.cpp)
