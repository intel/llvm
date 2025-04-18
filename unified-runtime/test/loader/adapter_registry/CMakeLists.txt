# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_adapter_reg_search_test name)
    set(TEST_TARGET_NAME ${name}-test)
    add_gtest_test(${name} ${TEST_SOURCES})

    if(WIN32)
        target_sources(${TEST_TARGET_NAME} PRIVATE
            ${PROJECT_SOURCE_DIR}/source/loader/windows/adapter_search.cpp)
    else()
        target_sources(${TEST_TARGET_NAME} PRIVATE
            ${PROJECT_SOURCE_DIR}/source/loader/linux/adapter_search.cpp)
    endif()

    target_include_directories(${TEST_TARGET_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/source/loader)
endfunction()

add_adapter_reg_search_test(search-with-env)
add_adapter_reg_search_test(search-no-env)
add_adapter_reg_search_test(search-order)
add_adapter_reg_search_test(prefilter)
