# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TEST_NAME example-collected-hello-world)

add_test(NAME ${TEST_NAME}
    COMMAND ${CMAKE_COMMAND}
    -D MODE=stdout
    -D TEST_FILE=$<TARGET_FILE:hello_world>
    -D MATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/hello_world.out.match
    -P ${PROJECT_SOURCE_DIR}/cmake/match.cmake
    DEPENDS collector hello_world
)

set_tests_properties(${TEST_NAME} PROPERTIES
    LABELS "tracing"
)

set_property(TEST ${TEST_NAME} PROPERTY ENVIRONMENT
    "XPTI_TRACE_ENABLE=1"
    "XPTI_FRAMEWORK_DISPATCHER=$<TARGET_FILE:xptifw>"
    "XPTI_SUBSCRIBERS=$<TARGET_FILE:collector>"
    "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_null>\""
    "UR_ENABLE_LAYERS=UR_LAYER_TRACING")
