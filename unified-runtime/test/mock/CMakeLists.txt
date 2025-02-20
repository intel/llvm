# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(MOCK_TEST_NAME test-mock)

add_ur_executable(${MOCK_TEST_NAME} mock.cpp)
target_link_libraries(${MOCK_TEST_NAME}
  PRIVATE
  ${PROJECT_NAME}::loader
  ${PROJECT_NAME}::headers
  ${PROJECT_NAME}::testing
  ${PROJECT_NAME}::mock
  GTest::gtest_main)

add_test(NAME ${MOCK_TEST_NAME}
    COMMAND ${MOCK_TEST_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(${MOCK_TEST_NAME} PROPERTIES LABELS "mock")
