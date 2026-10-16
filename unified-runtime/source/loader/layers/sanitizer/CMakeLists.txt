# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A shared library rather than a part of the loader, so that all loader
# instances of a process share one instance of it - see ur_shared_layer.hpp.
# Loaded on demand from the same locations an adapter is, and exports nothing
# but urLoaderLayerGetInterface, which keeps the LLVM symbolizer private too.

add_ur_library(ur_sanitizer_layer SHARED
    ${PROJECT_SOURCE_DIR}/source/ur/ur.cpp
    asan/asan_allocator.cpp
    asan/asan_buffer.cpp
    asan/asan_ddi.cpp
    asan/asan_interceptor.cpp
    asan/asan_quarantine.cpp
    asan/asan_report.cpp
    asan/asan_shadow.cpp
    asan/asan_statistics.cpp
    asan/asan_validator.cpp
    msan/msan_allocator.cpp
    msan/msan_buffer.cpp
    msan/msan_ddi.cpp
    msan/msan_interceptor.cpp
    msan/msan_report.cpp
    msan/msan_shadow.cpp
    tsan/tsan_buffer.cpp
    tsan/tsan_ddi.cpp
    tsan/tsan_interceptor.cpp
    tsan/tsan_report.cpp
    tsan/tsan_shadow.cpp
    sanitizer_common/linux/backtrace.cpp
    sanitizer_common/linux/sanitizer_utils.cpp
    sanitizer_common/sanitizer_allocator.cpp
    sanitizer_common/sanitizer_options.cpp
    sanitizer_common/sanitizer_stackdepot.cpp
    sanitizer_common/sanitizer_stacktrace.cpp
    sanitizer_common/sanitizer_utils.cpp
    ur_sanddi.cpp
    ur_sanitizer_layer.cpp
    ur_sanitizer_layer_entry.cpp
)
install_ur_library(ur_sanitizer_layer)

target_include_directories(ur_sanitizer_layer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/source
    ${PROJECT_SOURCE_DIR}/source/loader
    ${PROJECT_SOURCE_DIR}/source/loader/layers
)

target_link_libraries(ur_sanitizer_layer PRIVATE
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::headers
)

set_target_properties(ur_sanitizer_layer PROPERTIES
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    # BUILD_WITH_INSTALL_RPATH is intentionally not set, see
    # source/adapters/CMakeLists.txt.
    INSTALL_RPATH "\$ORIGIN"
)

target_link_options(ur_sanitizer_layer PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/sanitizer_layer.map"
)

if(UR_ENABLE_SYMBOLIZER)
    target_sources(ur_sanitizer_layer PRIVATE
        sanitizer_common/linux/symbolizer.cpp
    )
    target_compile_definitions(ur_sanitizer_layer PRIVATE UR_HAVE_SYMBOLIZER)
    target_include_directories(ur_sanitizer_layer PRIVATE ${LLVM_INCLUDE_DIRS})
    target_link_libraries(ur_sanitizer_layer PRIVATE LLVMSupport LLVMSymbolize)

    # The symbolizer dependencies may emit calls to
    # _intel_fast_memcpy/_intel_fast_memset, normally provided by libirc, which
    # this library doesn't link. The stubs forward to plain memcpy/memset.
    if(CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
        if(MSVC)
            set(_no_intel_lib_flag -Qno-intel-lib=libirc)
        else()
            set(_no_intel_lib_flag -no-intel-lib=libirc)
        endif()
        set_source_files_properties(
            ${PROJECT_SOURCE_DIR}/source/loader/intel_fast_mem_stub.c
            PROPERTIES COMPILE_OPTIONS "${_no_intel_lib_flag}")
        target_sources(ur_sanitizer_layer PRIVATE
            ${PROJECT_SOURCE_DIR}/source/loader/intel_fast_mem_stub.c
        )
    endif()

    # In an in-tree build, if LLVM is built with libc++, symbolizer.cpp has to
    # be built against libc++ as well and libc++ linked in.
    if(NOT UR_STANDALONE_BUILD AND LLVM_LIBCXX_USED)
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libc++.a
            OUTPUT_VARIABLE LIBCXX_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libc++abi.a
            OUTPUT_VARIABLE LIBCXX_ABI_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT EXISTS ${LIBCXX_PATH} OR NOT EXISTS ${LIBCXX_ABI_PATH})
            message(FATAL_ERROR "libc++ is required but can't find the libraries")
        endif()
        set_property(SOURCE
            ${CMAKE_CURRENT_SOURCE_DIR}/sanitizer_common/linux/symbolizer.cpp
            APPEND_STRING PROPERTY COMPILE_FLAGS " -stdlib=libc++ ")
        # Link with gcc_s first to avoid some symbols resolving to
        # libc++/libc++abi/libunwind's one
        target_link_libraries(ur_sanitizer_layer PRIVATE
            gcc_s ${LIBCXX_PATH} ${LIBCXX_ABI_PATH})
    endif()
endif()

# Loaded at runtime, so there is no link time dependency to pull it in.
add_dependencies(ur_loader ur_sanitizer_layer)

# MSan over the offload adapter: the layer dlopens libur_msan_ze.so, which
# liboffload's level_zero plugin then binds to as "libze_loader.so".
if(UR_BUILD_ADAPTER_OFFLOAD)
    include(FetchLevelZero)
    target_sources(ur_sanitizer_layer PRIVATE
        msan/msan_ze.cpp
        msan/msan_ze_intercept.cpp
    )
    # Headers only: the layer must not depend on libze_loader.
    target_link_libraries(ur_sanitizer_layer PRIVATE LevelZeroLoader-Headers)
    target_compile_definitions(ur_sanitizer_layer PRIVATE UR_MSAN_ZE=1)

    # Link-time stand-in giving the shim DT_NEEDED libze_loader.so.1 whether
    # LevelZeroLoader is shared or static. Not installed.
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/ze_loader_needed.cpp "")
    add_library(ur_msan_ze_needed SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/ze_loader_needed.cpp)
    set_target_properties(ur_msan_ze_needed PROPERTIES NO_SONAME ON)
    target_link_options(ur_msan_ze_needed PRIVATE
        "-Wl,-soname,libze_loader.so.1")

    add_ur_library(ur_msan_ze SHARED msan/ze_shim/msan_ze_shim.cpp)
    # Not in the export set: nothing links against it.
    install(TARGETS ur_msan_ze COMPONENT unified-runtime
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    target_link_libraries(ur_msan_ze PRIVATE
        LevelZeroLoader-Headers
        -Wl,--no-as-needed ur_msan_ze_needed -Wl,--as-needed
    )
    set_target_properties(ur_msan_ze PROPERTIES NO_SONAME ON)
    target_link_options(ur_msan_ze PRIVATE "-Wl,-soname,libze_loader.so")
    add_dependencies(ur_loader ur_msan_ze)
endif()
