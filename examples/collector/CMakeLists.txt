# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

set(TARGET_NAME collector)

add_library(${TARGET_NAME} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/collector.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME} PRIVATE xpti ${CMAKE_DL_LIBS})
target_include_directories(${TARGET_NAME} PRIVATE ${xpti_SOURCE_DIR}/include)

if(MSVC)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        XPTI_STATIC_LIBRARY XPTI_CALLBACK_API_EXPORTS)
endif()
