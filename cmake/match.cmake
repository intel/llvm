# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

#
# match.cmake -- script for creating ctests that compare the output of a binary
# with a known good match file.
#

find_package(Python3 COMPONENTS Interpreter)

if(NOT DEFINED TEST_FILE)
    message(FATAL_ERROR "TEST_FILE needs to be defined")
endif()
if(NOT DEFINED MATCH_FILE)
    message(FATAL_ERROR "MATCH_FILE needs to be defined")
endif()

set(TEST_OUT "_matchtmpfile")

if(NOT DEFINED TEST_ARGS) # easier than ifdefing the rest of the code
    set(TEST_ARGS "")
endif()

string(REPLACE "\"" "" TEST_ARGS "${TEST_ARGS}")
separate_arguments(TEST_ARGS)

execute_process(
    COMMAND ${TEST_FILE} ${TEST_ARGS}
    OUTPUT_FILE ${TEST_OUT}
    RESULT_VARIABLE TEST_RESULT
)

if(TEST_RESULT)
    message(FATAL_ERROR "Failed: Test ${TEST_FILE} ${TEST_ARGS} returned non-zero (${TEST_RESULT}).")
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/match.py ${TEST_OUT} ${MATCH_FILE}
    RESULT_VARIABLE TEST_RESULT
)

if(TEST_RESULT)
    message(FATAL_ERROR "Failed: The output of ${TEST_FILE} (stored in ${TEST_OUT}) does not match ${MATCH_FILE} (${TEST_RESULT})")
elseif()
    message("Passed: The output of ${TEST_FILE} matches ${MATCH_FILE}")
endif()
