# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_unit_test(logger
    logger.cpp
)

set(TEST_TARGET_NAME test-logger_env_var)
add_ur_executable(${TEST_TARGET_NAME}
    env_var.cpp
)
target_link_libraries(${TEST_TARGET_NAME}
    PRIVATE
    ${PROJECT_NAME}::common
    GTest::gtest_main
)

set(OUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/logger_test.log)

function(add_logger_env_var_log_match_test name env_var test_case match_file out)
    set(full_name unit-logger_env_var_${name})

    add_test(NAME ${full_name}-validate-output
        COMMAND ${CMAKE_COMMAND}
        -D MODE=${out}
        -D TEST_FILE=$<TARGET_FILE:${TEST_TARGET_NAME}>
        -D OUT_FILE=${OUT_FILE}
        -D MATCH_FILE=${match_file}
        -P ${PROJECT_SOURCE_DIR}/cmake/match.cmake
    )
    set_tests_properties(${full_name}-validate-output PROPERTIES
        LABELS "unit"
        ENVIRONMENT ${env_var}\;GTEST_FILTER=*${test_case}\;GTEST_BRIEF=1
    )
endfunction()

function(add_logger_env_var_no_logfile_test name env_var test_case)
    set(full_name unit-logger_env_var_${name})

    add_test(NAME ${full_name}-validate-no-file
        COMMAND ${CMAKE_COMMAND}
        -D BIN_PATH=${CMAKE_CURRENT_BINARY_DIR}/${TEST_TARGET_NAME}
        -D OUT_FILE=${OUT_FILE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/no_logfile.cmake
    )

    set_tests_properties(${full_name}-validate-no-file PROPERTIES
        LABELS "unit"
        ENVIRONMENT ${env_var}\;GTEST_FILTER=*${test_case}
    )
endfunction()

# file sink tests
add_logger_env_var_log_match_test(
    default_flush_lvl_err_msg
    UR_LOG_ADAPTER_TEST=level:error\\\\\;output:file,\"${OUT_FILE}\"
    ErrorMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_error_msg_exact.out.match
    "file"
)

add_logger_env_var_log_match_test(
    default_flush_lvl_no_output_warn
    UR_LOG_ADAPTER_TEST=level:error\\\\\;output:file,'${OUT_FILE}'
    WarningMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_empty.out.match
    "file"
)

add_logger_env_var_log_match_test(
    default_flush_lvl_all_lvls_msg
    UR_LOG_ADAPTER_TEST=level:debug\\\\\;output:file,'${OUT_FILE}'
    LoggerFromEnvVar*Message
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_all_levels_msg_exact.out.match
    "file"
)

# # stdout/stderr tests
add_logger_env_var_log_match_test(
    stdout_basic
    UR_LOG_ADAPTER_TEST=level:debug\\\\\;flush:debug\\\\\;output:stdout
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_debug_msg.out.match
    "stdout"
)

add_logger_env_var_log_match_test(
    stderr_basic
    UR_LOG_ADAPTER_TEST=level:debug\\\\\;flush:debug\\\\\;output:stderr
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_debug_msg_exact.out.match
    "stderr"
)

add_logger_env_var_log_match_test(
    stdout_with_default_lvls
    UR_LOG_ADAPTER_TEST=output:stdout
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_debug_msg.out.match
    "stdout"
)

add_logger_env_var_log_match_test(
    default_output_stderr_no_output
    UR_LOG_ADAPTER_TEST=flush:debug
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_empty.out.match
    "stderr"
)

add_logger_env_var_log_match_test(
    default_output_stderr_2
    UR_LOG_ADAPTER_TEST=level:debug
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_debug_msg_exact.out.match
    "stderr"
)

# log level tests
add_logger_env_var_log_match_test(
    default_lvl_no_output
    UR_LOG_ADAPTER_TEST=output:file,'${OUT_FILE}'
    WarningMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_empty.out.match
    "file"
)

add_logger_env_var_log_match_test(
    default_lvl_no_output_2
    UR_LOG_ADAPTER_TEST=flush:warning\\\\\;output:file,'${OUT_FILE}'
    WarningMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_empty.out.match
    "file"
)

add_logger_env_var_log_match_test(
    lvl_debug
    UR_LOG_ADAPTER_TEST=level:debug\\\\\;flush:debug\\\\\;output:file,'${OUT_FILE}'
    LoggerFromEnvVar*Message
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_all_levels_msg_exact.out.match
    "file"
)

add_logger_env_var_log_match_test(
    lvl_info
    UR_LOG_ADAPTER_TEST=level:info\\\\\;flush:info\\\\\;output:file,'${OUT_FILE}'
    InfoMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_info_msg_exact.out.match
    "file"
)

add_logger_env_var_log_match_test(
    lvl_info_err_msg
    UR_LOG_ADAPTER_TEST=level:info\\\\\;flush:info\\\\\;output:file,'${OUT_FILE}'
    ErrorMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_error_msg_exact.out.match
    "file"
)

add_logger_env_var_log_match_test(
    lvl_info_no_output
    UR_LOG_ADAPTER_TEST=level:info\\\\\;flush:info\\\\\;output:file,'${OUT_FILE}'
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_empty.out.match
    "file"
)

add_logger_env_var_log_match_test(
    lvl_warning
    UR_LOG_ADAPTER_TEST=level:warning\\\\\;flush:warning\\\\\;output:file,'${OUT_FILE}'
    WarningMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_warning_msg_exact.out.match
    "file"
)

add_logger_env_var_log_match_test(
    lvl_warning_err_msg
    UR_LOG_ADAPTER_TEST=level:warning\\\\\;flush:warning\\\\\;output:file,'${OUT_FILE}'
    ErrorMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_error_msg_exact.out.match
    "file"
)

add_logger_env_var_log_match_test(
    lvl_warning_no_output
    UR_LOG_ADAPTER_TEST=level:warning\\\\\;flush:warning\\\\\;output:file,'${OUT_FILE}'
    InfoMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_empty.out.match
    "file"
)

add_logger_env_var_log_match_test(
    lvl_error
    UR_LOG_ADAPTER_TEST=level:error\\\\\;flush:error\\\\\;output:file,'${OUT_FILE}'
    ErrorMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_error_msg_exact.out.match
    "file"
)

add_logger_env_var_log_match_test(
    lvl_error_no_output
    UR_LOG_ADAPTER_TEST=level:error\\\\\;flush:error\\\\\;output:file,'${OUT_FILE}'
    WarningMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_empty.out.match
    "file"
)

# wrong env var format - default logger, no output
add_logger_env_var_no_logfile_test(
    no_env_var
    UR_LOG=
    DebugMessage
)

add_logger_env_var_log_match_test(
    no_env_var
    UR_LOG=
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_debug_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    empty_env_var
    UR_LOG_ADAPTER_TEST=
    DebugMessage
)

add_logger_env_var_log_match_test(
    empty_env_var
    UR_LOG_ADAPTER_TEST=
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_debug_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    double_output
    UR_LOG_ADAPTER_TEST=output:stdout\\\\\;level:debug\\\\\;flush:debug\\\\\;output:file,'${OUT_FILE}'
    DebugMessage
)

add_logger_env_var_log_match_test(
    double_output
    UR_LOG_ADAPTER_TEST=output:stdout\\\\\;level:debug\\\\\;flush:debug\\\\\;output:file,'${OUT_FILE}'
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_debug_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    double_flush_level
    UR_LOG_ADAPTER_TEST=flush:error\\\\\;level:debug\\\\\;flush:debug\\\\\;output:file,'${OUT_FILE}'
    DebugMessage
)

add_logger_env_var_log_match_test(
    double_flush_level
    UR_LOG_ADAPTER_TEST=flush:error\\\\\;level:debug\\\\\;flush:debug\\\\\;output:file,'${OUT_FILE}'
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_debug_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    double_log_level
    UR_LOG_ADAPTER_TEST=level:error\\\\\;level:debug\\\\\;flush:debug\\\\\;output:file,'${OUT_FILE}'
    DebugMessage
)

add_logger_env_var_log_match_test(
    double_log_level
    UR_LOG_ADAPTER_TEST=level:error\\\\\;level:debug\\\\\;flush:debug\\\\\;output:file,'${OUT_FILE}'
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_debug_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    wrong_param
    UR_LOG_ADAPTER_TEST=lvl:debug\\\\\;flush:debug\\\\\;output:file,'${OUT_FILE}'
    DebugMessage
)

add_logger_env_var_log_match_test(
    wrong_param
    UR_LOG_ADAPTER_TEST=lvl:debug\\\\\;flush:debug\\\\\;output:file,'${OUT_FILE}'
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_debug_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    wrong_param_2
    UR_LOG_ADAPTER_TEST=lvl:debug
    DebugMessage
)

add_logger_env_var_log_match_test(
    wrong_param_2
    UR_LOG_ADAPTER_TEST=lvl:debug
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_debug_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    wrong_output
    UR_LOG_ADAPTER_TEST=level:error\\\\\;output:cout
    ErrorMessage
)

add_logger_env_var_log_match_test(
    wrong_output
    UR_LOG_ADAPTER_TEST=level:debug\\\\\;output:cout
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_debug_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    wrong_output_2
    UR_LOG_ADAPTER_TEST=level:error\\\\\;output:stdout,'${OUT_FILE}'
    ErrorMessage
)

add_logger_env_var_log_match_test(
    wrong_output_2
    UR_LOG_ADAPTER_TEST=level:debug\\\\\;output:stdout,'${OUT_FILE}'
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_debug_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    wrong_output_3
    UR_LOG_ADAPTER_TEST=level:error\\\\\;output:file
    ErrorMessage
)

add_logger_env_var_log_match_test(
    wrong_output_3
    UR_LOG_ADAPTER_TEST=level:debug\\\\\;output:file
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_debug_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    wrong_output_4
    UR_LOG_ADAPTER_TEST=level:error\\\\\;output:file,
    ErrorMessage
)

add_logger_env_var_log_match_test(
    wrong_output_4
    UR_LOG_ADAPTER_TEST=level:debug\\\\\;output:file,
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_debug_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    wrong_output_5
    UR_LOG_ADAPTER_TEST=level:error\\\\\;output:'${OUT_FILE}'
    ErrorMessage
)

add_logger_env_var_log_match_test(
    wrong_output_5
    UR_LOG_ADAPTER_TEST=level:debug\\\\\;output:'${OUT_FILE}'
    DebugMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_debug_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    wrong_level
    UR_LOG_ADAPTER_TEST=level:err\\\\\;output:file,'${OUT_FILE}'
    ErrorMessage
)

add_logger_env_var_log_match_test(
    wrong_level
    UR_LOG_ADAPTER_TEST=level:err\\\\\;output:file,'${OUT_FILE}'
    ErrorMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_error_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    wrong_flush_level
    UR_LOG_ADAPTER_TEST=level:error\\\\\;flush:err\\\\\;output:file,'${OUT_FILE}'
    ErrorMessage
)

add_logger_env_var_log_match_test(
    wrong_flush_level
    UR_LOG_ADAPTER_TEST=level:error\\\\\;flush:err\\\\\;output:file,'${OUT_FILE}'
    ErrorMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_error_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    random_str
    UR_LOG_ADAPTER_TEST=fvgwfe999&&@$
    ErrorMessage
)

add_logger_env_var_log_match_test(
    random_str
    UR_LOG_ADAPTER_TEST=fvgwfe999&&@$
    ErrorMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_error_msg.out.match
    "stdout"
)

add_logger_env_var_no_logfile_test(
    wrong_path
    UR_LOG_ADAPTER_TEST=level:error\\\\\;output:file,'${CMAKE_CURRENT_BINARY_DIR}/path/does/not/exist/logger_test.log'
    ErrorMessage
)

add_logger_env_var_log_match_test(
    wrong_path
    UR_LOG_ADAPTER_TEST=level:error\\\\\;output:file,'${CMAKE_CURRENT_BINARY_DIR}/path/does/not/exist/logger_test.log'
    ErrorMessage
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_no_error_msg.out.match
    "stdout"
)
