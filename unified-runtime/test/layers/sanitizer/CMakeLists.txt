# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_sanitizer_test name)
    add_gtest_test(${name} ${ARGN})
    target_include_directories(${name}-test PRIVATE
        ${PROJECT_SOURCE_DIR}/source
        ${PROJECT_SOURCE_DIR}/source/common
        ${PROJECT_SOURCE_DIR}/source/loader/layers/sanitizer/sanitizer_common
    )
endfunction()

function(add_test_source name source)
    target_sources(${name}-test PRIVATE ${source})
endfunction()

add_sanitizer_test(asan asan.cpp)
add_sanitizer_test(sanitizer_options sanitizer_options.cpp)

# add the source here in order to test for unexported functions
add_test_source(sanitizer_options 
    ${PROJECT_SOURCE_DIR}/source/loader/layers/sanitizer/sanitizer_common/sanitizer_options.cpp
)
