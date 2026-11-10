# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_adapter_reg_search_test name)
    set(TEST_TARGET_NAME ${name}-test)
    add_gtest_test(${name} ${ARGN})

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

add_adapter_reg_search_test(search-with-env search_with_env.cpp)
add_adapter_reg_search_test(search-no-env search_no_env.cpp)
add_adapter_reg_search_test(search-order search_order.cpp)
add_adapter_reg_search_test(prefilter prefilter.cpp)
