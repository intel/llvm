# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(hello_world)
if(UR_BUILD_EXAMPLE_CODEGEN)
    add_subdirectory(codegen)
endif()
if(UR_ENABLE_TRACING)
    add_subdirectory(collector)
endif()
