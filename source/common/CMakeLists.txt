# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_library(ur_common INTERFACE)
add_library(${PROJECT_NAME}::common ALIAS ur_common)

target_include_directories(ur_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

add_subdirectory(unified_memory_allocation)
target_link_libraries(ur_common INTERFACE unified_memory_allocation ${CMAKE_DL_LIBS} ${PROJECT_NAME}::headers)

target_sources(ur_common INTERFACE uma_helpers.hpp)

if(WIN32)
    target_sources(ur_common
        INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/windows/ur_lib_loader.cpp
            uma_helpers.hpp ur_pool_manager.hpp
    )
else()
    target_sources(ur_common
        INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/linux/ur_lib_loader.cpp
            uma_helpers.hpp ur_pool_manager.hpp
    )
endif()
