# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_adapter_reg_search_test name)
    cmake_parse_arguments(TEST "" "" "SEARCH_PATH;ENVS;SOURCES" ${ARGN})

    set(TEST_TARGET_NAME adapter-reg-test-${name})
    add_ur_executable(${TEST_TARGET_NAME}
        ${TEST_SOURCES})

    if(WIN32)
        target_sources(${TEST_TARGET_NAME} PRIVATE
            ${PROJECT_SOURCE_DIR}/source/loader/windows/adapter_search.cpp)
    else()
        target_sources(${TEST_TARGET_NAME} PRIVATE
            ${PROJECT_SOURCE_DIR}/source/loader/linux/adapter_search.cpp)
    endif()

    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::common
        GTest::gtest_main)

    target_include_directories(${TEST_TARGET_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/source/loader)

    add_test(NAME adapter-reg-${name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(adapter-reg-${name} PROPERTIES LABELS "adapter-reg")
    set_property(TEST adapter-reg-${name} PROPERTY
        ENVIRONMENT "UR_ADAPTERS_SEARCH_PATH=\"${TEST_SEARCH_PATH}\"" ${TEST_ENVS})
endfunction()

set(TEST_SEARCH_PATH ${PROJECT_SOURCE_DIR})
set(TEST_BIN_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
add_adapter_reg_search_test(search-with-env
    SEARCH_PATH ${TEST_SEARCH_PATH}
    ENVS "TEST_ADAPTER_SEARCH_PATH=\"${TEST_SEARCH_PATH}\"" "TEST_CUR_SEARCH_PATH=\"${TEST_BIN_PATH}\""
    SOURCES search_with_env.cpp)

add_adapter_reg_search_test(search-no-env
    SEARCH_PATH ""
    ENVS "TEST_ADAPTER_SEARCH_PATH=\"\"" "TEST_CUR_SEARCH_PATH=\"${TEST_BIN_PATH}\""
    SOURCES search_no_env.cpp)

add_adapter_reg_search_test(search-order
    SEARCH_PATH ${TEST_SEARCH_PATH}
    ENVS "TEST_ADAPTER_SEARCH_PATH=\"${TEST_SEARCH_PATH}\"" "TEST_CUR_SEARCH_PATH=\"${TEST_BIN_PATH}\""
    SOURCES search_order.cpp)

add_adapter_reg_search_test(prefilter
    SEARCH_PATH ""
    SOURCES prefilter.cpp)
