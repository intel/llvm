# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#
# helpers.cmake -- helper functions for top-level CMakeLists.txt
#

# Sets ${ret} to version of program specified by ${name} in major.minor format
function(get_program_version_major_minor name ret)
    execute_process(COMMAND ${name} --version
        OUTPUT_VARIABLE cmd_ret
        ERROR_QUIET)
    STRING(REGEX MATCH "([0-9]+)\.([0-9]+)" VERSION "${cmd_ret}")
    SET(${ret} ${VERSION} PARENT_SCOPE)
endfunction()

# Generates cppformat-$name targets and attaches them
# as dependencies of global "cppformat" target.
# Arguments are used as files to be checked.
# ${name} must be unique.
function(add_cppformat name)
    if(NOT CLANG_FORMAT OR NOT (CLANG_FORMAT_VERSION VERSION_EQUAL CLANG_FORMAT_REQUIRED))
        return()
    endif()

    if(${ARGC} EQUAL 0)
        return()
    else()
        # Split args into 2 parts (in Windows the list is probably too long)
        list(SUBLIST ARGN 0 250 selected_files_1)
        list(SUBLIST ARGN 251 -1 selected_files_2)
        add_custom_target(cppformat-${name}
            COMMAND ${CLANG_FORMAT} --style=file --i ${selected_files_1}
            COMMAND ${CLANG_FORMAT} --style=file --i ${selected_files_2}
            COMMENT "Format CXX source files"
        )
    endif()

    add_dependencies(cppformat cppformat-${name})
endfunction()

include(CheckCXXCompilerFlag)

macro(add_sanitizer_flag flag)
    set(SAVED_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
    set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES} -fsanitize=${flag}")

    check_cxx_compiler_flag("-fsanitize=${flag}" CXX_HAS_SANITIZER)
    if(CXX_HAS_SANITIZER)
        add_compile_options(-fsanitize=${flag})
        add_link_options(-fsanitize=${flag})
    else()
        message("${flag} sanitizer not supported")
    endif()

    set(CMAKE_REQUIRED_LIBRARIES ${SAVED_CMAKE_REQUIRED_LIBRARIES})
endmacro()

if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    check_cxx_compiler_flag("-fcf-protection=full" CXX_HAS_FCF_PROTECTION_FULL)
    check_cxx_compiler_flag("-fstack-clash-protection" CXX_HAS_FSTACK_CLASH_PROTECTION)
endif()

if (UR_USE_CFI AND UR_USE_ASAN)
    message(WARNING "Both UR_USE_CFI and UR_USE_ASAN are ON. "
        "Due to build errors, this is unsupported; CFI checks will be disabled")
    set(UR_USE_CFI OFF)
endif()

if (UR_USE_CFI)
    set(SAVED_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS "-flto -fvisibility=hidden")
    check_cxx_compiler_flag("-fsanitize=cfi" CXX_HAS_CFI_SANITIZE)
    set(CMAKE_REQUIRED_FLAGS ${SAVED_CMAKE_REQUIRED_FLAGS})
else()
    # If CFI checking is disabled, pretend we don't support it
    set(CXX_HAS_CFI_SANITIZE OFF)
endif()

set(CFI_FLAGS "")
if (CFI_HAS_CFI_SANITIZE)
    # cfi-icall requires called functions in shared libraries to also be built with cfi-icall, which we can't
    # guarantee. -fsanitize=cfi depends on -flto
    set(CFI_FLAGS "-flto -fsanitize=cfi -fno-sanitize=cfi-icall -fsanitize-ignorelist=${CMAKE_SOURCE_DIR}/sanitizer-ignorelist.txt")
endif()

function(add_ur_target_compile_options name)
    if(NOT MSVC)
        target_compile_definitions(${name} PRIVATE -D_FORTIFY_SOURCE=2)
        target_compile_options(${name} PRIVATE
            # Warning options
            -Wall
            -Wpedantic
            -Wempty-body
            -Wformat
            -Wformat-security
            -Wunused-parameter

            # Hardening options
            -fPIC
            -fstack-protector-strong
            -fvisibility=hidden

            ${CFI_FLAGS}
            $<$<BOOL:${CXX_HAS_FCF_PROTECTION_FULL}>:-fcf-protection=full>
            $<$<BOOL:${CXX_HAS_FSTACK_CLASH_PROTECTION}>:-fstack-clash-protection>

            # Colored output
            $<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>
            $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fcolor-diagnostics>
        )
        if (UR_DEVELOPER_MODE)
            target_compile_options(${name} PRIVATE -Werror -Wextra)
        endif()
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(${name} PRIVATE -fvisibility=hidden)
        endif()
    elseif(MSVC)
        target_compile_options(${name} PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/MP>  # clang-cl.exe does not support /MP
            /W3
            /MD$<$<CONFIG:Debug>:d>
            /GS
            /DWIN32_LEAN_AND_MEAN
            /DNOMINMAX
        )

        if(UR_DEVELOPER_MODE)
            # _CRT_SECURE_NO_WARNINGS used mainly because of getenv
            # C4267: The compiler detected a conversion from size_t to a smaller type.
            target_compile_options(${name} PRIVATE
                /WX /GS /D_CRT_SECURE_NO_WARNINGS /wd4267
            )
        endif()
    endif()
endfunction()

function(add_ur_target_link_options name)
    if(NOT MSVC)
        if (NOT APPLE)
            target_link_options(${name} PRIVATE
                ${CFI_FLAGS}
                "LINKER:-z,relro,-z,now,-z,noexecstack"
            )
            if (UR_DEVELOPER_MODE)
                target_link_options(${name} PRIVATE -Werror -Wextra)
            endif()
            if (CMAKE_BUILD_TYPE STREQUAL "Release")
                target_link_options(${name} PRIVATE
                    $<$<CXX_COMPILER_ID:GNU>:-pie>
                )
            endif()
        endif()
    elseif(MSVC)
        target_link_options(${name} PRIVATE
            LINKER:/DYNAMICBASE
            LINKER:/HIGHENTROPYVA
            LINKER:/NXCOMPAT
        )
    endif()
endfunction()

function(add_ur_target_exec_options name)
    if(MSVC)
        target_link_options(${name} PRIVATE
            LINKER:/ALLOWISOLATION
        )
    endif()
endfunction()

function(add_ur_executable name)
    add_executable(${name} ${ARGN})
    add_ur_target_compile_options(${name})
    add_ur_target_exec_options(${name})
    add_ur_target_link_options(${name})
endfunction()

function(add_ur_library name)
    add_library(${name} ${ARGN})
    add_ur_target_compile_options(${name})
    add_ur_target_link_options(${name})
    if(MSVC)
        target_link_options(${name} PRIVATE
            $<$<STREQUAL:$<TARGET_LINKER_FILE_NAME:${name}>,link.exe>:LINKER:/DEPENDENTLOADFLAG:0x2000>
        )
    endif()
endfunction()

function(install_ur_library name)
    install(TARGETS ${name}
            EXPORT ${PROJECT_NAME}-targets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT unified-runtime
    )
endfunction()

include(FetchContent)

function(FetchSource GIT_REPOSITORY GIT_TAG GIT_DIR DEST)
    message(STATUS "Fetching sparse source ${GIT_DIR} from ${GIT_REPOSITORY} ${GIT_TAG}")
    IF(NOT EXISTS ${DEST})
        file(MAKE_DIRECTORY ${DEST})
        execute_process(COMMAND git init
            WORKING_DIRECTORY ${DEST})
        execute_process(COMMAND git checkout -b main
            WORKING_DIRECTORY ${DEST})
        execute_process(COMMAND git remote add origin ${GIT_REPOSITORY}
            WORKING_DIRECTORY ${DEST})
        execute_process(COMMAND git config core.sparsecheckout true
            WORKING_DIRECTORY ${DEST})
        file(APPEND ${DEST}/.git/info/sparse-checkout ${GIT_DIR}/)
    endif()
    execute_process(COMMAND git fetch --depth=1 origin refs/tags/${GIT_TAG}:refs/tags/${GIT_TAG}
        WORKING_DIRECTORY ${DEST})
    execute_process(COMMAND git checkout --quiet ${GIT_TAG}
        WORKING_DIRECTORY ${DEST})
endfunction()

# A wrapper around FetchContent_Declare that supports git sparse checkout.
# This is useful for including subprojects from large repositories.
function(FetchContentSparse_Declare name GIT_REPOSITORY GIT_TAG GIT_DIR)
    set(content-build-dir ${CMAKE_BINARY_DIR}/content-${name})
    FetchSource(${GIT_REPOSITORY} ${GIT_TAG} ${GIT_DIR} ${content-build-dir})
    FetchContent_Declare(${name} SOURCE_DIR ${content-build-dir}/${GIT_DIR})
endfunction()
