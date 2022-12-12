# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT

add_library(common INTERFACE)
add_library(${PROJECT_NAME}::common ALIAS common)

target_include_directories(common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
