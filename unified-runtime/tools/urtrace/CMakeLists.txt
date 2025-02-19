# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TARGET_NAME ur_collector)

add_ur_library(${TARGET_NAME} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/collector.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME} PRIVATE ${TARGET_XPTI} ${PROJECT_NAME}::common ${CMAKE_DL_LIBS})
target_include_directories(${TARGET_NAME} PRIVATE ${xpti_SOURCE_DIR}/include)

if(MSVC)
    target_compile_definitions(${TARGET_NAME} PRIVATE XPTI_STATIC_LIBRARY)
endif()
target_compile_definitions(${TARGET_NAME} PRIVATE XPTI_CALLBACK_API_EXPORTS)

set(UR_TRACE_CLI_BIN ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/urtrace)

add_custom_target(ur_trace_cli)
add_custom_command(TARGET ur_trace_cli PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/urtrace.py ${UR_TRACE_CLI_BIN})
add_dependencies(ur_collector ur_trace_cli)
