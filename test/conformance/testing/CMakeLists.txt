# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_library(ur_testing STATIC 
    source/utils.cpp)
target_include_directories(ur_testing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ur_testing PRIVATE gtest_main unified-runtime::headers)
add_library(${PROJECT_NAME}::testing ALIAS ur_testing)
