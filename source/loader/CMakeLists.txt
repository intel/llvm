# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/UrLoaderVersion.rc.in
    ${CMAKE_CURRENT_BINARY_DIR}/UrLoaderVersion.rc
    @ONLY
)

add_ur_library(ur_loader
    SHARED
    ""
    ${CMAKE_CURRENT_BINARY_DIR}/UrLoaderVersion.rc
)

set_target_properties(ur_loader PROPERTIES
    LIBRARY_OUTPUT_NAME ur_loader
    RUNTIME_OUTPUT_NAME ur_loader
)

add_library(${PROJECT_NAME}::loader ALIAS ur_loader)

target_include_directories(ur_loader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/layers
)

set_target_properties(ur_loader PROPERTIES
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
)

target_link_libraries(ur_loader PRIVATE
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::headers
)

if(UR_ENABLE_TRACING)
    target_link_libraries(ur_loader PRIVATE ${TARGET_XPTI})
    target_include_directories(ur_loader PRIVATE ${xpti_SOURCE_DIR}/include)
    target_compile_definitions(ur_loader PRIVATE XPTI_STATIC_LIBRARY)
endif()

if (UNIX)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    target_link_libraries(ur_loader PRIVATE Threads::Threads)
    # Older gcc versions need -pthread, not just -lpthread
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.1)
            if(CMAKE_USE_PTHREADS_INIT)
                target_link_libraries(ur_loader PRIVATE "-pthread")
            endif()
        endif()
    endif()
endif()

if(WIN32)
    target_link_libraries(ur_loader PRIVATE cfgmgr32.lib)
endif()

if(UNIX)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/libur_loader.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/libur_loader.pc
        @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libur_loader.pc" DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig" COMPONENT unified-runtime)
endif()

install(TARGETS ur_loader
    EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT unified-runtime
)

target_sources(ur_loader
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_object.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_loader.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_loader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_ldrddi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_ldrddi.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_libapi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_libddi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_lib.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_lib.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_codeloc.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_print.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/validation/ur_valddi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/validation/ur_validation_layer.cpp
)

if(UR_ENABLE_TRACING)
    target_sources(ur_loader
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/tracing/ur_tracing_layer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/tracing/ur_trcddi.cpp
    )
endif()

if(UR_ENABLE_SANITIZER)
    target_sources(ur_loader
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../ur/ur.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/asan_interceptor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/asan_interceptor.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/device_sanitizer_report.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/stacktrace.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/stacktrace.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/ur_sanddi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/ur_sanitizer_layer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/ur_sanitizer_layer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/linux/symbolizer_llvm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/symbolizer_llvm.hpp
    )

    target_sources(ur_loader
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/linux/san_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer/linux/backtrace.cpp
    )

    target_include_directories(ur_loader PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/layers/sanitizer"
        "${CMAKE_CURRENT_SOURCE_DIR}/../"
    )
endif()


# link validation backtrace dependencies
if(UNIX)
    find_package(Libbacktrace)
endif()
if (VAL_USE_LIBBACKTRACE_BACKTRACE AND LIBBACKTRACE_FOUND)
    message(STATUS "Using libbacktrace backtrace for validation")

    target_sources(ur_loader PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/validation/backtrace_libbacktrace.cpp)
    target_link_libraries(ur_loader PRIVATE Libbacktrace)
else()
    message(STATUS "Using default backtrace for validation")

    if(WIN32)
        target_sources(ur_loader PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/layers/validation/backtrace_win.cpp)
        target_link_libraries(ur_loader PRIVATE dbghelp)
    else()
        target_sources(ur_loader PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/layers/validation/backtrace_lin.cpp)
    endif()
endif()

if(WIN32)
    target_sources(ur_loader
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/windows/adapter_search.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/windows/lib_init.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/windows/loader_init.cpp
    )
else()
    target_sources(ur_loader
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/linux/adapter_search.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/linux/lib_init.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/linux/loader_init.cpp
    )
endif()
