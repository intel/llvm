# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

set(TEST_NAME example-collected-hello-world)

add_test(NAME ${TEST_NAME}
    COMMAND ${CMAKE_COMMAND}
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
    "UR_ADAPTERS_FORCE_LOAD=$<TARGET_FILE:ur_adapter_null>")
