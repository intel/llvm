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

function(set_tracing_test_props target_name collector_name)
    set_tests_properties(${target_name} PROPERTIES
        LABELS "tracing"
    )

    set_property(TEST ${target_name} PROPERTY ENVIRONMENT
        "XPTI_TRACE_ENABLE=1"
        "XPTI_FRAMEWORK_DISPATCHER=$<TARGET_FILE:xptifw>"
        "XPTI_SUBSCRIBERS=$<TARGET_FILE:${collector_name}>"
        "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_mock>\""
        "UR_ENABLE_LAYERS=UR_LAYER_TRACING")
endfunction()

add_test(NAME example-collected-hello-world
    COMMAND ${CMAKE_COMMAND}
    -D MODE=stdout
    -D TEST_FILE=$<TARGET_FILE:hello_world>
    -D MATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/hello_world.out.match
    -P ${PROJECT_SOURCE_DIR}/cmake/match.cmake
)

set_tracing_test_props(example-collected-hello-world collector)

add_test(NAME example-logged-hello-world
    COMMAND ${CMAKE_COMMAND}
    -D MODE=stdout
    -D TEST_FILE=$<TARGET_FILE:hello_world>
    -D MATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/hello_world.out.logged.match
    -P ${PROJECT_SOURCE_DIR}/cmake/match.cmake
)
set_tests_properties(example-logged-hello-world PROPERTIES LABELS "tracing")
set_property(TEST example-logged-hello-world PROPERTY ENVIRONMENT
    "UR_LOG_TRACING=level:info\;output:stdout"
    "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_mock>\""
    "UR_ENABLE_LAYERS=UR_LAYER_TRACING")

function(add_tracing_test name)
    set(TEST_TARGET_NAME tracing-test-${name})
    add_ur_executable(${TEST_TARGET_NAME}
        ${ARGN})
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        ${PROJECT_NAME}::loader
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::testing
        GTest::gtest_main)
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND}
        -D MODE=stderr
        -D TEST_FILE=$<TARGET_FILE:${TEST_TARGET_NAME}>
        -D MATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}.out.match
        -P ${PROJECT_SOURCE_DIR}/cmake/match.cmake
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tracing_test_props(${name} test_collector)
endfunction()

add_tracing_test(codeloc codeloc.cpp)
