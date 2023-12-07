# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TEST_NAME example-sanitized-codegen)

add_test(NAME ${TEST_NAME}
    COMMAND ${CMAKE_COMMAND}
    -D MODE=stdout
    -D TEST_FILE=$<TARGET_FILE:codegen>
    -D MATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/codegen.out.match
    -P ${PROJECT_SOURCE_DIR}/cmake/match.cmake
    DEPENDS codegen
)

set_tests_properties(${TEST_NAME} PROPERTIES
    LABELS "asan"
)

set_property(TEST ${TEST_NAME} PROPERTY ENVIRONMENT
    "UR_LOG_SANITIZER=level:debug"
    "UR_ENABLE_LAYERS=UR_LAYER_ASAN")
