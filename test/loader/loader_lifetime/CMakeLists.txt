# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_executable(test-loader-lifetime
    urLoaderInit.cpp
    urLoaderTearDown.cpp
)

target_link_libraries(test-loader-lifetime
    PRIVATE
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::loader
    gmock
    GTest::gtest_main
)

add_test(NAME loader-lifetime
    COMMAND test-loader-lifetime
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
