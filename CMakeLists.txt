# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)
project(unified-runtime VERSION 0.5.0)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(helpers)

find_package(Python3 COMPONENTS Interpreter)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build Options
option(UR_DEVELOPER_MODE "enable developer checks, treats warnings as errors" OFF)
option(UR_BUILD_TESTS "Build unit tests." ON)
option(UR_FORMAT_CPP_STYLE "format code style of C++ sources" OFF)
option(UR_USE_ASAN "enable AddressSanitizer" OFF)
option(UR_USE_UBSAN "enable UndefinedBehaviorSanitizer" OFF)
option(UR_USE_MSAN "enable MemorySanitizer" OFF)
option(UMA_BUILD_SHARED_LIBRARY "Build UMA as shared library" OFF)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(MSVC)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)
endif()

# Define rpath for libraries so that adapters can be found automatically
set(CMAKE_BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

# Define a path for custom commands to work around MSVC
set(CUSTOM_COMMAND_BINARY_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
if(MSVC)
    # MSVC implicitly adds $<CONFIG> to the output path
    set(CUSTOM_COMMAND_BINARY_DIR ${CUSTOM_COMMAND_BINARY_DIR}/$<CONFIG>)
endif()

# CXX flags setup
if(NOT MSVC)
    add_compile_options(-fPIC -Wall -Wpedantic)
    if(UR_DEVELOPER_MODE)
        add_compile_options(-fno-omit-frame-pointer)
        add_compile_options(-Werror)
    endif()
elseif(MSVC)
    string(REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/MDd" "/MTd" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/MD" "/MT" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    add_compile_options(/MP /W3)

    if(UR_DEVELOPER_MODE)
        add_compile_options(/WX)
    endif()
endif()

if(UR_USE_ASAN)
    add_sanitizer_flag(address)
endif()

if(UR_USE_UBSAN)
    add_sanitizer_flag(undefined)
endif()

if(UR_USE_MSAN)
    message(WARNING "MemorySanitizer requires that all code is built with
        its instrumentation, otherwise false positives are possible.
        See https://github.com/google/sanitizers/wiki/MemorySanitizerLibcxxHowTo#instrumented-libc
        for details")
    add_sanitizer_flag(memory)
endif()

# Allow custom third_party folder
if(NOT DEFINED THIRD_PARTY_DIR)
    set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
endif()

find_program(CLANG_FORMAT NAMES clang-format-15 clang-format-15.0 clang-format)

set(CLANG_FORMAT_REQUIRED "15.0")

if(CLANG_FORMAT)
    get_program_version_major_minor(${CLANG_FORMAT} CLANG_FORMAT_VERSION)
    message(STATUS "Found clang-format: ${CLANG_FORMAT} (version: ${CLANG_FORMAT_VERSION})")
endif()

# Check if the correct version of clang-format is available
if(UR_FORMAT_CPP_STYLE)
    if(CLANG_FORMAT)
        if(NOT (CLANG_FORMAT_VERSION VERSION_EQUAL CLANG_FORMAT_REQUIRED))
            message(FATAL_ERROR "required clang-format version is ${CLANG_FORMAT_REQUIRED}")
        endif()
    else()
        message(FATAL_ERROR "UR_FORMAT_CPP_STYLE=ON, but clang-format not found (required version: ${CLANG_FORMAT_REQUIRED})")
    endif()
endif()

# Obtain files for clang-format
file(GLOB_RECURSE format_src
    "*.h"
    "*.hpp"
    "*.cpp"
)

# Exclude files from build directory to ensure they're not passed to the formatter
list(FILTER format_src EXCLUDE REGEX "${CMAKE_CURRENT_BINARY_DIR}.")

# Add code formatter target
add_custom_target(cppformat)

# Add files to the formatter
add_cppformat(src-formatter ${format_src})

add_subdirectory(${THIRD_PARTY_DIR})

# A header only library to specify include directories in transitive
# dependencies.
add_library(headers INTERFACE)
# Alias target to support FetchContent.
add_library(${PROJECT_NAME}::headers ALIAS headers)
target_include_directories(headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Add the include directory and the headers target to the install.
install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION include COMPONENT headers)
install(
    TARGETS headers
    EXPORT ${PROJECT_NAME}-targets)

add_subdirectory(source)
add_subdirectory(examples)
if(UR_BUILD_TESTS)
    add_subdirectory(test)
endif()

# Add the list of installed targets to the install. This includes the namespace
# which all installed targets will be prefixed with, e.g. for the headers
# target users will depend on ${PROJECT_NAME}::headers.
install(
    EXPORT ${PROJECT_NAME}-targets
    FILE ${PROJECT_NAME}-targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME})

# Configure the package versions file for use in find_package when installed.
write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}-config-version.cmake
    COMPATIBILITY SameMajorVersion)
# Configure the package file that is searched for by find_package when
# installed.
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in
    ${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME})

# Add the package files to the install.
install(
    FILES
        ${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}-config.cmake
        ${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}-config-version.cmake
    DESTINATION lib/cmake/${PROJECT_NAME})

set(API_JSON_FILE ${PROJECT_BINARY_DIR}/unified_runtime.json)

# Generate source from the specification
add_custom_target(generate-code USES_TERMINAL
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/scripts
    COMMAND ${Python3_EXECUTABLE} run.py --api-json ${API_JSON_FILE}
    COMMAND ${Python3_EXECUTABLE} json2src.py --api-json ${API_JSON_FILE} ${PROJECT_SOURCE_DIR}
)

# Generate and format source from the specification
add_custom_target(generate USES_TERMINAL
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target generate-code
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target cppformat
)

# Generate source and check for uncommitted diffs
add_custom_target(check-generated
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND git diff --exit-code
    DEPENDS generate
)
