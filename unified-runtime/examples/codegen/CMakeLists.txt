# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TARGET_NAME codegen)

find_package(LLVM CONFIG)
find_package(PkgConfig)

set(TRANSLATOR_FOUND "FALSE")
if(${PkgConfig_FOUND})
    pkg_check_modules(LLVMSPIRVLib IMPORTED_TARGET LLVMSPIRVLib)
endif()

if(LLVM_FOUND AND PkgConfig_FOUND AND LLVMSPIRVLib_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    llvm_map_components_to_libnames(llvm_libs support core irreader bitwriter)

    add_ur_executable(${TARGET_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/codegen.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers.cpp
    )

    target_include_directories(${TARGET_NAME} PRIVATE ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(${TARGET_NAME} PRIVATE ${LLVM_DEFINITIONS})
    target_link_libraries(${TARGET_NAME}
        PRIVATE
            ${CMAKE_DL_LIBS}
            ${PROJECT_NAME}::headers
            ${PROJECT_NAME}::loader
            LLVM
            PkgConfig::LLVMSPIRVLib
    )
    # TODO: Depend on building adapters.

    if(MSVC)
        set_target_properties(${TARGET_NAME}
            PROPERTIES
                VS_DEBUGGER_COMMAND_ARGUMENTS ""
                VS_DEBUGGER_WORKING_DIRECTORY "$(OutDir)"
        )
    endif()
else()
    message(FATAL_ERROR "The environment did not satisfy dependency requirements (LLVM, PkgConfig, LLVMSPIRVLib) for codegen example (skipping target).")
endif()
