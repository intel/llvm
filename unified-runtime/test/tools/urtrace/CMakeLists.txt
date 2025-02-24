# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TEST_NAME trace-hello-world)

function(add_trace_test name CLI_ARGS)
    set(TEST_NAME trace_test_${name})
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${name}.match
        ${CMAKE_CURRENT_BINARY_DIR}/${name}.match
        @ONLY
    )
    add_test(NAME ${TEST_NAME}
        COMMAND ${CMAKE_COMMAND}
        -D TEST_FILE=${Python3_EXECUTABLE}
        -D TEST_ARGS="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/urtrace --stdout ${CLI_ARGS} --flush info $<TARGET_FILE:hello_world>"
        -D MODE=stdout
        -D MATCH_FILE=${CMAKE_CURRENT_BINARY_DIR}/${name}.match
        -P ${PROJECT_SOURCE_DIR}/cmake/match.cmake
    )
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS "urtrace")
endfunction()

add_trace_test(mock_hello "--libpath $<TARGET_FILE_DIR:ur_adapter_mock> --mock")
add_trace_test(mock_hello_no_args "--libpath $<TARGET_FILE_DIR:ur_adapter_mock> --mock --no-args")
add_trace_test(mock_hello_filter_device "--libpath $<TARGET_FILE_DIR:ur_adapter_mock> --mock --filter \".*Device.*\"")
add_trace_test(mock_hello_profiling "--libpath $<TARGET_FILE_DIR:ur_adapter_mock> --mock --profiling --time-unit ns")
add_trace_test(mock_hello_begin "--libpath $<TARGET_FILE_DIR:ur_adapter_mock> --mock --print-begin")
add_trace_test(mock_hello_json "--libpath $<TARGET_FILE_DIR:ur_adapter_mock> --mock --json")
