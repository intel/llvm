# Copyright (C) 2023 Intel Corporation
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
        add_custom_target(cppformat-${name}
            COMMAND ${CLANG_FORMAT}
                --style=file
                --i
                ${ARGN}
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

function(add_ur_target_compile_options name)
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE
            -fPIC
            -Wall
            -Wpedantic
            -Wempty-body
            -Wunused-parameter
            $<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>
            $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fcolor-diagnostics>
        )
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_definitions(${name} PRIVATE -D_FORTIFY_SOURCE=2)
        endif()
        if(UR_DEVELOPER_MODE)
            target_compile_options(${name} PRIVATE
                -Werror
                -fno-omit-frame-pointer
                -fstack-protector-strong
            )
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
            target_link_options(${name} PRIVATE "LINKER:-z,relro,-z,now")
        endif()
    elseif(MSVC)
        target_link_options(${name} PRIVATE
            /DYNAMICBASE
            /HIGHENTROPYVA
            /NXCOMPAT
        )
    endif()
endfunction()

function(add_ur_target_exec_options name)
    if(MSVC)
        target_link_options(${name} PRIVATE
            /ALLOWISOLATION
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
