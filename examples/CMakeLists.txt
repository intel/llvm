# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(hello_world)
add_subdirectory(codegen)
if(UR_ENABLE_TRACING)
    add_subdirectory(collector)
endif()
