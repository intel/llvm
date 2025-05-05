# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_library(ur_testing STATIC 
    source/utils.cpp
    source/fixtures.cpp)
target_include_directories(ur_testing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ur_testing PRIVATE
    gtest_main
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::headers)
add_library(${PROJECT_NAME}::testing ALIAS ur_testing)
