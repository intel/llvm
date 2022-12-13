# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT

set(TARGET_NAME hello_world)

add_executable(${TARGET_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_world.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
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
)
