# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_library(ur_mock SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/ur_mock_helpers.cpp"
)
target_include_directories(ur_mock
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(ur_mock PRIVATE
    ${PROJECT_NAME}::headers
)

add_library(${PROJECT_NAME}::mock ALIAS ur_mock)
