# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(UR_UMF_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_ur_library(umf_test_helpers STATIC
    ${UR_UMF_TEST_DIR}/common/pool.c
    ${UR_UMF_TEST_DIR}/common/provider.c
)

target_link_libraries(umf_test_helpers PRIVATE ${PROJECT_NAME}::common)

function(add_umf_test name)
    set(TEST_TARGET_NAME umf_test-${name})
    add_ur_executable(${TEST_TARGET_NAME}
        ${ARGN})
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        ${PROJECT_NAME}::unified_malloc_framework
        ${PROJECT_NAME}::common
        umf_test_helpers
        GTest::gtest_main)
    target_include_directories(${TEST_TARGET_NAME} PRIVATE
        ${UR_UMF_TEST_DIR}/common)
    add_test(NAME  umf-${name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(umf-${name} PROPERTIES LABELS "umf")
    if (UMF_ENABLE_POOL_TRACKING)
        target_compile_definitions(${TEST_TARGET_NAME} PRIVATE UMF_ENABLE_POOL_TRACKING_TESTS=1)
    endif()
endfunction()

add_subdirectory(umf_pools)

add_umf_test(memoryProvider memoryProviderAPI.cpp)
add_umf_test(memoryPool memoryPoolAPI.cpp)
add_umf_test(base base.cpp)
