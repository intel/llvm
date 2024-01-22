# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)
project(unified-runtime VERSION 0.9.0)

include(GNUInstallDirs)
include(CheckCXXSourceCompiles)
include(CMakePackageConfigHelpers)
include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(helpers)

if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(Python3_FIND_FRAMEWORK NEVER)
    set(Python3_FIND_STRATEGY LOCATION)
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Build Options
option(UR_BUILD_TESTS "Build unit tests." ON)
option(UR_BUILD_TOOLS "build ur tools" ON)
option(UR_FORMAT_CPP_STYLE "format code style of C++ sources" OFF)
option(UR_DEVELOPER_MODE "enable developer checks, treats warnings as errors" OFF)
option(UR_USE_ASAN "enable AddressSanitizer" OFF)
option(UR_USE_UBSAN "enable UndefinedBehaviorSanitizer" OFF)
option(UR_USE_MSAN "enable MemorySanitizer" OFF)
option(UR_USE_TSAN "enable ThreadSanitizer" OFF)
option(UR_ENABLE_TRACING "enable api tracing through xpti" OFF)
option(UR_ENABLE_SANITIZER "enable device sanitizer" ON)
option(UMF_BUILD_SHARED_LIBRARY "Build UMF as shared library" OFF)
option(UMF_ENABLE_POOL_TRACKING "Build UMF with pool tracking" ON)
option(UR_BUILD_ADAPTER_L0 "Build the Level-Zero adapter" OFF)
option(UR_BUILD_ADAPTER_OPENCL "Build the OpenCL adapter" OFF)
option(UR_BUILD_ADAPTER_CUDA "Build the CUDA adapter" OFF)
option(UR_BUILD_ADAPTER_HIP "Build the HIP adapter" OFF)
option(UR_BUILD_ADAPTER_NATIVE_CPU "Build the Native-CPU adapter" OFF)
option(UR_BUILD_ADAPTER_ALL "Build all currently supported adapters" OFF)
option(UR_BUILD_EXAMPLE_CODEGEN "Build the codegen example." OFF)
option(VAL_USE_LIBBACKTRACE_BACKTRACE "enable libbacktrace validation backtrace for linux" OFF)
set(UR_DPCXX "" CACHE FILEPATH "Path of the DPC++ compiler executable")
set(UR_SYCL_LIBRARY_DIR "" CACHE PATH
    "Path of the SYCL runtime library directory")
option(UR_ENABLE_ASSERTIONS "Enable assertions for all build types" OFF)

include(Assertions)

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

if(NOT MSVC)
    # Determine if libstdc++ is being used.
    check_cxx_source_compiles("
        #include <array>
        #ifndef __GLIBCXX__
        #error not using libstdc++
        #endif
        int main() {}"
        USING_LIBSTDCXX)
    if(USING_LIBSTDCXX)
        # Support older versions of GCC where the <filesystem> header is not
        # available and <experimental/filesystem> must be used instead. This
        # requires linking against libstdc++fs.a, on systems where <filesystem>
        # is available we still need to link this library.
        link_libraries(stdc++fs)
    endif()
endif()

if(UR_ENABLE_TRACING)
    add_compile_definitions(UR_ENABLE_TRACING)

    # fetch xpti proxy library for the tracing layer
    FetchContentSparse_Declare(xpti https://github.com/intel/llvm.git "sycl-nightly/20230703" "xpti")
    FetchContent_MakeAvailable(xpti)

    # set -fPIC for xpti since we are linking it with a shared library
    set_target_properties(xpti PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # fetch the xptifw dispatcher, mostly used for testing
    # these variables need to be set for xptifw to compile
    set(XPTI_SOURCE_DIR ${xpti_SOURCE_DIR})
    set(XPTI_DIR ${xpti_SOURCE_DIR})
    set(XPTI_ENABLE_TESTS OFF CACHE INTERNAL "Turn off xptifw tests")

    FetchContentSparse_Declare(xptifw https://github.com/intel/llvm.git "sycl-nightly/20230703" "xptifw")

    FetchContent_MakeAvailable(xptifw)

    check_cxx_compiler_flag("-Wno-error=maybe-uninitialized" HAS_MAYBE_UNINIT)
    if (HAS_MAYBE_UNINIT)
        target_compile_options(xptifw PRIVATE -Wno-error=maybe-uninitialized)
    endif()

    set_target_properties(xptifw PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
    if (MSVC)
        set(TARGET_XPTI $<IF:$<CONFIG:Release>,xpti,xptid>)
    else()
        set(TARGET_XPTI xpti)
    endif()
endif()

if(UR_ENABLE_SANITIZER)
    add_compile_definitions(UR_ENABLE_SANITIZER)
endif()

if(UR_USE_ASAN)
    add_sanitizer_flag(address)
endif()

if(UR_USE_UBSAN)
    add_sanitizer_flag(undefined)
endif()

if(UR_USE_TSAN)
    add_sanitizer_flag(thread)
endif()

if(UR_USE_MSAN)
    message(WARNING "MemorySanitizer requires that all code is built with
        its instrumentation, otherwise false positives are possible.
        See https://github.com/google/sanitizers/wiki/MemorySanitizerLibcxxHowTo#instrumented-libc
        for details")
    add_sanitizer_flag(memory)
endif()

# Check if clang-format (in correct version) is available for Cpp code formatting.
if(UR_FORMAT_CPP_STYLE)
    find_program(CLANG_FORMAT NAMES clang-format-15 clang-format-15.0 clang-format)

    if(CLANG_FORMAT)        
        get_program_version_major_minor(${CLANG_FORMAT} CLANG_FORMAT_VERSION)
        message(STATUS "Found clang-format: ${CLANG_FORMAT} (version: ${CLANG_FORMAT_VERSION})")

        set(CLANG_FORMAT_REQUIRED "15.0")
        if(NOT (CLANG_FORMAT_VERSION VERSION_EQUAL CLANG_FORMAT_REQUIRED))
            message(FATAL_ERROR "required clang-format version is ${CLANG_FORMAT_REQUIRED}")
        endif()
    else()
        message(FATAL_ERROR "UR_FORMAT_CPP_STYLE=ON, but clang-format not found (required version: ${CLANG_FORMAT_REQUIRED})")
    endif()
endif()

# Obtain files for clang-format and license check
set(format_glob)
set(license_glob)
foreach(dir examples include source test tools)
    list(APPEND format_glob
        "${dir}/*.h"
        "${dir}/*.hpp"
        "${dir}/*.c"
        "${dir}/*.cpp"
        "${dir}/**/*.h"
        "${dir}/**/*.hpp"
        "${dir}/**/*.c"
        "${dir}/**/*.cpp")
    list(APPEND license_glob
        "${dir}/*.yml"
        "${dir}/**/*.yml"
        "${dir}/*.py"
        "${dir}/**/*.py"
        "${dir}/**/CMakeLists.txt"
        "${dir}/CMakeLists.txt"
    )
endforeach()
file(GLOB_RECURSE format_src ${format_glob})
file(GLOB_RECURSE license_src ${license_glob})

# Add license check target
list(FILTER license_src EXCLUDE REGEX "registry.yml")
add_custom_target(verify-licenses
    COMMAND ${Python3_EXECUTABLE}
        "${PROJECT_SOURCE_DIR}/scripts/verify_license.py"
        "--files" ${format_src} ${license_src}
    COMMENT "Verify all files contain a license."
)

# Add code formatter target
add_custom_target(cppformat)
# ... and all source files to the formatter
add_cppformat(all-sources ${format_src})

# Allow custom third_party folder
if(NOT DEFINED THIRD_PARTY_DIR)
    set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
endif()

add_subdirectory(${THIRD_PARTY_DIR})

# A header only library to specify include directories in transitive
# dependencies.
add_library(ur_headers INTERFACE)
# Alias target to support FetchContent.
add_library(${PROJECT_NAME}::headers ALIAS ur_headers)
target_include_directories(ur_headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Add the include directory and the headers target to the install.
install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION include COMPONENT ur_headers)
install(
    TARGETS ur_headers
    EXPORT ${PROJECT_NAME}-targets)

add_subdirectory(source)
add_subdirectory(examples)
if(UR_BUILD_TESTS)
    add_subdirectory(test)
endif()
if(UR_BUILD_TOOLS)
    add_subdirectory(tools)
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

if(UR_FORMAT_CPP_STYLE)
    # Generate source from the specification
    add_custom_target(generate-code USES_TERMINAL
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/scripts
        COMMAND ${Python3_EXECUTABLE} run.py --api-json ${API_JSON_FILE} --clang-format=${CLANG_FORMAT}
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
else()
    message(STATUS "  UR_FORMAT_CPP_STYLE not set. Targets: 'generate' and 'check-generated' are not available")
endif()
