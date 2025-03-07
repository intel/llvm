# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_library(test_collector SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/test_collector.cpp
)

target_include_directories(test_collector PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_collector PRIVATE ${TARGET_XPTI})
target_include_directories(test_collector PRIVATE ${xpti_SOURCE_DIR}/include)

if(MSVC)
    target_compile_definitions(test_collector PRIVATE XPTI_STATIC_LIBRARY)
endif()
target_compile_definitions(test_collector PRIVATE XPTI_CALLBACK_API_EXPORTS)

function(add_tracing_test name)
    set(TEST_TARGET_NAME tracing-test-${name})
    add_ur_executable(${TEST_TARGET_NAME}
        ${ARGN})
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        ${PROJECT_NAME}::loader
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::testing
        GTest::gtest_main)
endfunction()

add_tracing_test(codeloc codeloc.cpp)
add_dependencies(check-unified-runtime tracing-test-codeloc hello_world collector test_collector xptifw)
