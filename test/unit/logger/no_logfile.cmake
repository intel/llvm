# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#
# no_logfile.cmake -- wrapper script for tests verifying that log file was not created
#

if(NOT DEFINED BIN_PATH)
    message(FATAL_ERROR "BIN_PATH needs to be defined with a test binary to be ran")
endif()
if(NOT DEFINED OUT_FILE)
    message(FATAL_ERROR "OUT_FILE needs to be defined with an output file to be verified")
endif()

if(EXISTS ${OUT_FILE})
    file(REMOVE ${OUT_FILE})
endif()

execute_process(
    COMMAND ${BIN_PATH}
)

if(EXISTS ${OUT_FILE})
    file(REMOVE ${OUT_FILE})
    message(FATAL_ERROR "Failed: File ${OUT_FILE} should not be created")
else()
    message("Passed: File ${OUT_FILE} does not exist")
endif()
