# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

set(TEST_NAME trace-hello-world)

function(add_trace_test name CLI_ARGS)
    set(TEST_NAME trace_test_${name})
    add_test(NAME ${TEST_NAME}
        COMMAND ${CMAKE_COMMAND}
        -D TEST_FILE=${Python3_EXECUTABLE}
        -D TEST_ARGS="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/urtrace --stdout ${CLI_ARGS} $<TARGET_FILE:hello_world>"
        -D MODE=stdout
        -D MATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}.match
        -P ${PROJECT_SOURCE_DIR}/cmake/match.cmake
        DEPENDS ur_trace_cli hello_world
    )
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS "urtrace")
endfunction()

add_trace_test(null_hello "--libpath $<TARGET_FILE_DIR:ur_adapter_null> --null")
add_trace_test(null_hello_no_args "--libpath $<TARGET_FILE_DIR:ur_adapter_null> --null --no-args")
add_trace_test(null_hello_filter_device "--libpath $<TARGET_FILE_DIR:ur_adapter_null> --null --filter \".*Device.*\"")
add_trace_test(null_hello_profiling "--libpath $<TARGET_FILE_DIR:ur_adapter_null> --null --profiling --time-unit ns")
add_trace_test(null_hello_begin "--libpath $<TARGET_FILE_DIR:ur_adapter_null> --null --print-begin")
add_trace_test(null_hello_json "--libpath $<TARGET_FILE_DIR:ur_adapter_null> --null --json")
