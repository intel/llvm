# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(NATIVE_CPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Native CPU adapter directory")

set(TARGET_NAME ur_adapter_native_cpu)

add_ur_adapter(${TARGET_NAME}
        SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/command_buffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/context.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/enqueue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/image.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/memory.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nativecpu_state.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/platform.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/platform.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/program.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/program.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/queue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/queue.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sampler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_interface_loader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/usm_p2p.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/virtual_mem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/usm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.hpp
)

set_target_properties(${TARGET_NAME} PROPERTIES
        VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
        SOVERSION "${PROJECT_VERSION_MAJOR}"
)

find_package(Threads REQUIRED)

target_link_libraries(${TARGET_NAME} PRIVATE
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::common
        Threads::Threads
)

target_include_directories(${TARGET_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../"
)
