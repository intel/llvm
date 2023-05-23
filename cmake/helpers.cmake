# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

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

include(FetchContent)

# A wrapper around FetchContent_Declare that supports git sparse checkout.
# This is useful for including subprojects from large repositories.
function(FetchContentSparse_Declare name GIT_REPOSITORY GIT_TAG GIT_DIR)
    set(content-build-dir ${CMAKE_BINARY_DIR}/content-${name})
    message(STATUS "Fetching sparse content ${GIT_DIR} from ${GIT_REPOSITORY} ${GIT_TAG}")
    IF(NOT EXISTS ${content-build-dir})
        file(MAKE_DIRECTORY ${content-build-dir})
        execute_process(COMMAND git init -b main
            WORKING_DIRECTORY ${content-build-dir})
        execute_process(COMMAND git remote add origin ${GIT_REPOSITORY}
            WORKING_DIRECTORY ${content-build-dir})
        execute_process(COMMAND git config core.sparsecheckout true
            WORKING_DIRECTORY ${content-build-dir})
        file(APPEND ${content-build-dir}/.git/info/sparse-checkout ${GIT_DIR}/)
    endif()
    execute_process(COMMAND git pull --depth=1 origin ${GIT_TAG}
        WORKING_DIRECTORY ${content-build-dir})
    FetchContent_Declare(${name} SOURCE_DIR ${content-build-dir}/${GIT_DIR})
endfunction()
