# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TARGET_NAME collector)

add_ur_library(${TARGET_NAME} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/collector.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME} PRIVATE ${TARGET_XPTI})
target_include_directories(${TARGET_NAME} PRIVATE ${xpti_SOURCE_DIR}/include)

if(MSVC)
    target_compile_definitions(${TARGET_NAME} PRIVATE XPTI_STATIC_LIBRARY)
endif()
target_compile_definitions(${TARGET_NAME} PRIVATE XPTI_CALLBACK_API_EXPORTS)
