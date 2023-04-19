# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(UR_ENABLE_TRACING)
    add_subdirectory(urtrace)
endif()
