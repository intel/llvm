# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TARGET_NAME hello_world)

add_ur_executable(${TARGET_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_world.cpp
)

if(MSVC)
    set_target_properties(${TARGET_NAME}
        PROPERTIES
            VS_DEBUGGER_COMMAND_ARGUMENTS ""
            VS_DEBUGGER_WORKING_DIRECTORY "$(OutDir)"
    )
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    ${PROJECT_NAME}::loader
    ${CMAKE_DL_LIBS}
    ${PROJECT_NAME}::headers
)
