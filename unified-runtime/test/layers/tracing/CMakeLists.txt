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

add_gtest_test(codeloc codeloc.cpp)
