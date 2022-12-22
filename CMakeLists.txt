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

# CXX compiler support
if(NOT MSVC)
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
    CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
    if(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    elseif(COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support.  Please use a different C++ compiler.")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -fPIC")
endif()

# MSVC compile flags
if(MSVC)
    string(REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

    # Treat warnings as errors
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX /W3 /wd4996")

    # Enable multi-process compilation
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

# Allow custom third_party folder
if(NOT DEFINED THIRD_PARTY_DIR)
    set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
endif()

# Option to disable tests
option(${PROJECT_NAME}_BUILD_TESTS "Build unit tests." ON)

# Option to run a code formatter
option(${PROJECT_NAME}_FORMAT_CPP_STYLE "format code style of C++ sources" OFF)

find_program(CLANG_FORMAT NAMES clang-format-10 clang-format-10.0 clang-format)

set(CLANG_FORMAT_REQUIRED "10.0")

if(CLANG_FORMAT)
    get_program_version_major_minor(${CLANG_FORMAT} CLANG_FORMAT_VERSION)
    message(STATUS "Found clang-format: ${CLANG_FORMAT} (version: ${CLANG_FORMAT_VERSION})")
endif()

# Check if the correct version of clang-format is available
if(FORMAT_CPP_STYLE)
    if(CLANG_FORMAT)
        if(NOT (CLANG_FORMAT_VERSION VERSION_EQUAL CLANG_FORMAT_REQUIRED))
            message(FATAL_ERROR "required clang-format version is ${CLANG_FORMAT_REQUIRED}")
        endif()
    else()
        message(FATAL_ERROR "FORMAT_CPP_STYLE=ON, but clang-format not found (required version: ${CLANG_FORMAT_REQUIRED})")
    endif()
endif()

# Obtain files for clang-format
file(GLOB_RECURSE format_src
    "*.h"
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
if(UR_ENABLE_TESTS)
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
add_custom_target(generate
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/scripts
    COMMAND ${Python3_EXECUTABLE} run.py --api-json ${API_JSON_FILE}
    COMMAND ${Python3_EXECUTABLE} json2src.py --api-json ${API_JSON_FILE} ${PROJECT_SOURCE_DIR}
)

# Generate source and check for uncommitted diffs
add_custom_target(check-generated
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND git diff --exit-code
    DEPENDS generate
    # Uncomment line below after docker image is integrated
    #DEPENDS cppformat
)
