# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(NATIVE_CPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Native CPU adapter directory")

set(TARGET_NAME ur_adapter_native_cpu)

add_ur_adapter(${TARGET_NAME}
        SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/async_alloc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/command_buffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/context.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/enqueue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/image.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/memory_export.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/memory.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nativecpu_state.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/platform.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/platform.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/program.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/program.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/queue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/queue.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sampler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/threadpool.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_interface_loader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/usm_p2p.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/virtual_mem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/usm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.hpp
)
install_ur_library(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES
        VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
        SOVERSION "${PROJECT_VERSION_MAJOR}"
)

# oneTBB is used as an optional NativeCPU backend and disabled by default.
option(NATIVECPU_WITH_ONETBB "Use oneTBB as backend for Native CPU" OFF)
if(NATIVECPU_WITH_ONETBB)
  message(STATUS "Configuring Native CPU adapter with oneTBB backend.")

  include(FetchContent)
  FetchContent_Declare(
    tbb
    GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
# commit 4e4fffed4fb86ae0960a3364700f549b539c777e (HEAD -> master, origin/master, origin/HEAD)
# Author: Ilya Isaev <ilya.isaev@intel.com>
# Date:   Mon Aug 18 10:35:26 2025 +0200
# Improve task_arena interoperability with task_groups (#1784)
    GIT_TAG 4e4fffed4fb86ae0960a3364700f549b539c777e
    CMAKE_ARGS "-DTBB_TEST:BOOL=OFF -DTBB_EXAMPLES:BOOL=OFF -DTBB_BENCH:BOOL=OFF"
    OVERRIDE_FIND_PACKAGE
  )
  set(TBB_TEST OFF CACHE INTERNAL "" FORCE)
  set(TBB_EXAMPLES OFF CACHE INTERNAL "" FORCE)
  set(TBB_BENCH OFF CACHE INTERNAL "" FORCE)
  set(TBB_BUILD ON CACHE INTERNAL "" FORCE)
  set(TBB_FIND_PACKAGE OFF CACHE INTERNAL "" FORCE)
  set(TBB_FUZZ_TESTING OFF CACHE INTERNAL "" FORCE)
  set(TBB_INSTALL ON CACHE INTERNAL "" FORCE)
  set (CMAKE_INCLUDE_CURRENT_DIR OFF)
  FetchContent_MakeAvailable(tbb)
endif()

find_package(Threads REQUIRED)

target_link_libraries(${TARGET_NAME} PRIVATE
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::common
        ${PROJECT_NAME}::umf
        Threads::Threads
)

target_include_directories(${TARGET_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../"
)

if(NATIVECPU_WITH_ONETBB)
  target_link_libraries(${TARGET_NAME} PRIVATE
        TBB::tbb
  )
  if (NOT MSVC)
    # oneTBB currently casts away some const qualifiers
    # todo: check if compiler actually supports these options
    target_compile_options(tbb PRIVATE -Wno-cast-qual -Wno-stringop-overflow -Wno-unknown-warning-option)
    target_compile_options(tbbmalloc PRIVATE -Wno-cast-qual)
  endif()

  # Undefine _DEBUG option in release builds to find
  # release tbbbind
  if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    target_compile_options(tbb PRIVATE -U_DEBUG)
  endif()

  target_compile_definitions(${TARGET_NAME} PRIVATE NATIVECPU_WITH_ONETBB)
endif()
