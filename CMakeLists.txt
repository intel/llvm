# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)
project(unified-runtime VERSION 0.12.0)

# Check if unified runtime is built as a standalone project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR UR_STANDALONE_BUILD)
    set(UR_STANDALONE_BUILD TRUE)
endif()

include(GNUInstallDirs)
include(CheckCXXSourceCompiles)
include(CMakePackageConfigHelpers)
include(CTest)

# Build Options
option(UR_BUILD_EXAMPLES "Build example applications." ON)
option(UR_BUILD_TESTS "Build unit tests." ON)
option(UR_BUILD_TOOLS "build ur tools" ON)
option(UR_FORMAT_CPP_STYLE "format code style of C++ sources" OFF)
option(UR_DEVELOPER_MODE "treats warnings as errors" OFF)
option(UR_ENABLE_FAST_SPEC_MODE "enable fast specification generation mode" OFF)
option(UR_USE_ASAN "enable AddressSanitizer" OFF)
option(UR_USE_UBSAN "enable UndefinedBehaviorSanitizer" OFF)
option(UR_USE_MSAN "enable MemorySanitizer" OFF)
option(UR_USE_TSAN "enable ThreadSanitizer" OFF)
option(UR_USE_CFI "enable Control Flow Integrity checks (requires clang and implies -flto)" OFF)
option(UR_ENABLE_TRACING "enable api tracing through xpti" OFF)
option(UR_ENABLE_SANITIZER "enable device sanitizer" ON)
option(UR_ENABLE_SYMBOLIZER "enable symoblizer for sanitizer" OFF)
option(UMF_BUILD_SHARED_LIBRARY "Build UMF as shared library" ON)
option(UMF_ENABLE_POOL_TRACKING "Build UMF with pool tracking" ON)
option(UR_BUILD_ADAPTER_L0 "Build the Level-Zero adapter" OFF)
option(UR_BUILD_ADAPTER_OPENCL "Build the OpenCL adapter" OFF)
option(UR_BUILD_ADAPTER_CUDA "Build the CUDA adapter" OFF)
option(UR_BUILD_ADAPTER_HIP "Build the HIP adapter" OFF)
option(UR_BUILD_ADAPTER_NATIVE_CPU "Build the Native-CPU adapter" OFF)
option(UR_BUILD_ADAPTER_ALL "Build all currently supported adapters" OFF)
option(UR_BUILD_ADAPTER_L0_V2 "Build the (experimental) Level-Zero v2 adapter" OFF)
option(UR_STATIC_ADAPTER_L0 "Build the Level-Zero adapter as static and embed in the loader" OFF)
option(UR_BUILD_EXAMPLE_CODEGEN "Build the codegen example." OFF)
option(VAL_USE_LIBBACKTRACE_BACKTRACE "enable libbacktrace validation backtrace for linux" OFF)
option(UR_ENABLE_ASSERTIONS "Enable assertions for all build types" OFF)
option(UR_BUILD_XPTI_LIBS "Build the XPTI libraries when tracing is enabled" ON)
option(UR_STATIC_LOADER "Build loader as a static library" OFF)
option(UR_FORCE_LIBSTDCXX "Force use of libstdc++ in a build using libc++ on Linux" OFF)
option(UR_ENABLE_LATENCY_HISTOGRAM "Enable latncy histogram" OFF)
set(UR_DPCXX "" CACHE FILEPATH "Path of the DPC++ compiler executable")
set(UR_DPCXX_BUILD_FLAGS "" CACHE STRING "Build flags to pass to DPC++ when compiling device programs")
set(UR_SYCL_LIBRARY_DIR "" CACHE PATH
    "Path of the SYCL runtime library directory")
set(UR_CONFORMANCE_TARGET_TRIPLES "" CACHE STRING
    "List of sycl targets to build CTS device binaries for")
set(UR_CONFORMANCE_AMD_ARCH "" CACHE STRING "AMD device target ID to build CTS binaries for")
option(UR_CONFORMANCE_ENABLE_MATCH_FILES "Enable CTS match files" ON)
option(UR_CONFORMANCE_TEST_LOADER "Also test the loader in the conformance tests" OFF)
set(UR_ADAPTER_LEVEL_ZERO_SOURCE_DIR "" CACHE PATH
    "Path to external 'level_zero' adapter source dir")
set(UR_ADAPTER_OPENCL_SOURCE_DIR "" CACHE PATH
    "Path to external 'opencl' adapter source dir")
set(UR_ADAPTER_CUDA_SOURCE_DIR "" CACHE PATH
    "Path to external 'cuda' adapter source dir")
set(UR_ADAPTER_HIP_SOURCE_DIR "" CACHE PATH
    "Path to external 'hip' adapter source dir")
set(UR_ADAPTER_NATIVE_CPU_SOURCE_DIR "" CACHE PATH
    "Path to external 'native_cpu' adapter source dir")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(helpers)

if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(Python3_FIND_FRAMEWORK NEVER)
    set(Python3_FIND_STRATEGY LOCATION)
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# There's little reason not to generate the compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(Assertions)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Define rpath for libraries so that adapters can be found automatically
set(CMAKE_BUILD_RPATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

# Define a path for custom commands to work around MSVC
set(CUSTOM_COMMAND_BINARY_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
if(CMAKE_SYSTEM_NAME STREQUAL Windows AND NOT CMAKE_GENERATOR STREQUAL Ninja)
    # MSVC implicitly adds $<CONFIG> to the output path
    set(CUSTOM_COMMAND_BINARY_DIR ${CUSTOM_COMMAND_BINARY_DIR}/$<CONFIG>)
endif()

if(UR_FORCE_LIBSTDCXX AND CMAKE_SYSTEM_NAME STREQUAL Linux)
    # Remove flags to specify using libc++ or static libstdc++ in order to
    # support sitatuions where the libstdc++ ABI is required.
    foreach(flags CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
        string(REPLACE "-stdlib=libc++" "" ${flags} "${${flags}}")
        string(REPLACE "-static-libstdc++" "" ${flags} "${${flags}}")
    endforeach()
    # Globally link against pthread, this is necessary when forcing use of
    # libstdc++ in a libc++ build as the FindThreads module may have already
    # been invoked and detected that pthread symbols are provided by libc++
    # which is not the case for libstdc++.
    add_compile_options(-pthread)
    link_libraries(pthread)
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
    if(UR_FORCE_LIBSTDCXX OR USING_LIBSTDCXX)
        # Support older versions of GCC where the <filesystem> header is not
        # available and <experimental/filesystem> must be used instead. This
        # requires linking against libstdc++fs.a, on systems where <filesystem>
        # is available we still need to link this library.
        link_libraries(stdc++fs)
    endif()
endif()

if(UR_ENABLE_TRACING)
    add_compile_definitions(UR_ENABLE_TRACING)

    if (UR_BUILD_XPTI_LIBS)
        # fetch xpti proxy library for the tracing layer
        FetchContentSparse_Declare(xpti https://github.com/intel/llvm.git "nightly-2024-10-22" "xpti")
        FetchContent_MakeAvailable(xpti)

        # set -fPIC for xpti since we are linking it with a shared library
        set_target_properties(xpti PROPERTIES POSITION_INDEPENDENT_CODE ON)

        # fetch the xptifw dispatcher, mostly used for testing
        # these variables need to be set for xptifw to compile
        set(XPTI_SOURCE_DIR ${xpti_SOURCE_DIR})
        set(XPTI_DIR ${xpti_SOURCE_DIR})
        set(XPTI_ENABLE_TESTS OFF CACHE INTERNAL "Turn off xptifw tests")

        FetchContentSparse_Declare(xptifw https://github.com/intel/llvm.git "nightly-2024-10-22" "xptifw")

        FetchContent_MakeAvailable(xptifw)

        check_cxx_compiler_flag("-Wno-error=maybe-uninitialized" HAS_MAYBE_UNINIT)
        if (HAS_MAYBE_UNINIT)
            target_compile_options(xptifw PRIVATE -Wno-error=maybe-uninitialized)
        endif()

        set_target_properties(xptifw PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        )

        if (NOT MSVC)
            # Hardening flags cause issues on Windows
            add_ur_target_compile_options(xptifw)
            add_ur_target_link_options(xptifw)
        endif()

        if (UR_STATIC_LOADER)
            install(TARGETS xpti xptifw
                EXPORT ${PROJECT_NAME}-targets
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            )
        endif()
    endif()

    if (MSVC)
        set(TARGET_XPTI $<IF:$<CONFIG:Release>,xpti,xptid>)
    else()
        set(TARGET_XPTI xpti)
    endif()
endif()

if(UR_ENABLE_SANITIZER)
    if(APPLE)
        message(WARNING "Sanitizer layer isn't supported on macOS")
        set(UR_ENABLE_SANITIZER OFF)
    elseif(WIN32)
        message(WARNING "Sanitizer layer isn't supported on Windows")
        set(UR_ENABLE_SANITIZER OFF)
    else()
        add_compile_definitions(UR_ENABLE_SANITIZER)
    endif()

    if(UR_ENABLE_SYMBOLIZER AND UR_STANDALONE_BUILD)
        find_package(LLVM REQUIRED)
    endif()
else()
    if(UR_ENABLE_SYMBOLIZER)
        message(FATAL_ERROR "Symbolizer must be enabled with Sanitizer layer")
    endif()
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

if(NOT (UR_BUILD_ADAPTER_CUDA OR UR_BUILD_ADAPTER_HIP
    OR UR_BUILD_ADAPTER_L0 OR UR_BUILD_ADAPTER_OPENCL
    OR UR_BUILD_ADAPTER_NATIVE_CPU OR UR_BUILD_ADAPTER_L0_V2
    OR UR_BUILD_ADAPTER_ALL))
    message(WARNING "No adapters have been enabled; conformance tests will not be ran")
    message(STATUS "Consider setting UR_BUILD_ADAPTER_*")
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

# Add hardening check
add_custom_target(verify-hardening
    COMMAND "${PROJECT_SOURCE_DIR}/scripts/check-hardening.sh"
        ${CMAKE_BINARY_DIR}
    COMMENT "Check hardening settings on built binaries and libraries"
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
if(UR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
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
        COMMAND ${Python3_EXECUTABLE} run.py
            --api-json ${API_JSON_FILE}
            --clang-format=${CLANG_FORMAT}
            $<$<BOOL:${UR_ENABLE_FAST_SPEC_MODE}>:--fast-mode>
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
