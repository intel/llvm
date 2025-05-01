# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(configure_targets_header input output)
    cmake_parse_arguments(PARSE_ARGV 2 ARG "" VAR_PREFIX TARGETS)
    if (ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unexpected arguments \"${ARG_UNPARSED_ARGUMENTS}\"")
    endif()

    list(LENGTH ARG_TARGETS "${ARG_VAR_PREFIX}_NUM")
    list(TRANSFORM ARG_TARGETS PREPEND "\"")
    list(TRANSFORM ARG_TARGETS APPEND "\"")
    list(JOIN ARG_TARGETS ",\n" "${ARG_VAR_PREFIX}_INITIALIZER")

    configure_file(${input} ${output} @ONLY)
endfunction()

configure_targets_header(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/uur/targets.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/uur/targets.h
    VAR_PREFIX UR_CONFORMANCE_TARGET_TRIPLES
    TARGETS ${TARGET_TRIPLES})

add_ur_library(ur_testing STATIC 
    source/utils.cpp
    source/fixtures.cpp)
target_include_directories(ur_testing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include)
target_link_libraries(ur_testing PRIVATE
    gtest_main
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::headers)
add_library(${PROJECT_NAME}::testing ALIAS ur_testing)
