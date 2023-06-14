# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TARGET_NAME codegen)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
llvm_map_components_to_libnames(llvm_libs support core irreader bitwriter)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LLVMSPIRVLib REQUIRED IMPORTED_TARGET LLVMSPIRVLib)

add_executable(${TARGET_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/codegen.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(${TARGET_NAME} PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(${TARGET_NAME} PRIVATE ${LLVM_DEFINITIONS})
target_link_libraries(${TARGET_NAME} PRIVATE PkgConfig::LLVMSPIRVLib LLVM)

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
