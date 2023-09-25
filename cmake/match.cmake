# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#
# match.cmake -- script for creating ctests that compare the output of a binary
# with a known good match file.
#

find_package(Python3 COMPONENTS Interpreter)

# Process variables passed to the script
if(NOT DEFINED MODE)
    message(FATAL_ERROR "MODE needs to be defined. Possible values: 'stdout', 'stderr', 'file'")
elseif(${MODE} STREQUAL "stdout" OR ${MODE} STREQUAL "stderr")
    if(NOT DEFINED TEST_FILE)
        message(FATAL_ERROR "TEST_FILE needs to be defined with a path to a binary to run")
    else()
        set(OUT_FILE "_matchtmpfile")
    endif()
elseif(${MODE} STREQUAL "file")
    if(NOT DEFINED OUT_FILE)
        message(FATAL_ERROR "OUT_FILE needs to be defined with an output file to be verified")
    endif()
else()
    message(FATAL_ERROR "${MODE} mode not recognised. Possible values: 'stdout', 'stderr', 'file'")
endif()
if(NOT DEFINED MATCH_FILE)
    message(FATAL_ERROR "MATCH_FILE needs to be defined")
endif()
if(NOT DEFINED TEST_ARGS) # easier than ifdefing the rest of the code
    set(TEST_ARGS "")
endif()

string(REPLACE "\"" "" TEST_ARGS "${TEST_ARGS}")
separate_arguments(TEST_ARGS)

if(EXISTS ${OUT_FILE})
    file(REMOVE ${OUT_FILE})
endif()

# Run the test binary. Capture the output to the temporary file.
if(${MODE} STREQUAL "stdout")
    execute_process(
        COMMAND ${TEST_FILE} ${TEST_ARGS}
        OUTPUT_FILE ${OUT_FILE}
        RESULT_VARIABLE TEST_RESULT
    )
elseif(${MODE} STREQUAL "stderr")
    execute_process(
        COMMAND ${TEST_FILE} ${TEST_ARGS}
        ERROR_FILE ${OUT_FILE}
        RESULT_VARIABLE TEST_RESULT
    )
elseif(${MODE} STREQUAL "file")
    execute_process(
        COMMAND ${TEST_FILE}
        RESULT_VARIABLE TEST_RESULT
    )
endif()

if(TEST_RESULT)
    message(FATAL_ERROR "Failed: Test ${TEST_FILE} ${TEST_ARGS} returned non-zero (${TEST_RESULT}).")
endif()

# Compare the output file contents with a match file contents
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/match.py ${OUT_FILE} ${MATCH_FILE}
    RESULT_VARIABLE TEST_RESULT
)

if(TEST_RESULT)
    message(FATAL_ERROR "Failed (${TEST_RESULT}): The output of ${OUT_FILE} does not match ${MATCH_FILE}")
elseif()
    message("Passed: The output ${OUT_FILE} matches ${MATCH_FILE}")
endif()

if(EXISTS ${OUT_FILE})
    file(REMOVE ${OUT_FILE})
endif()
