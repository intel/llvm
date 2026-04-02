# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TARGET_NAME ur_adapter_offload)

add_ur_adapter(${TARGET_NAME}
    SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sampler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ur2offload.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ur_interface_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/usm.cpp
)

set(UR_OFFLOAD_INSTALL_DIR "" CACHE PATH "Path to the directory containing libomptarget.so etc")
set(UR_OFFLOAD_INCLUDE_DIR "" CACHE PATH "Path to the directory containing LLVM headers")
if (UR_OFFLOAD_INSTALL_DIR STREQUAL "" OR UR_OFFLOAD_INCLUDE_DIR STREQUAL "")
    include(ExternalProject)
    set(LLVM_PROJECT_SOURCE_DIR  ${CMAKE_BINARY_DIR}/llvm-src-offload)
    set(LLVM_PROJECT_TAG  32beea0605f37ea7a6429375d41b19ee78ddfe7d)
    set(OPENMP_INSTALL_DIR ${CMAKE_BINARY_DIR}/openmp-install)
    set(UR_OFFLOAD_INSTALL_DIR ${CMAKE_BINARY_DIR}/offload-install)
    set(UR_OFFLOAD_INCLUDE_DIR ${UR_OFFLOAD_INSTALL_DIR}/include)

    execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" worktree prune)

    if(NOT IS_DIRECTORY "${LLVM_PROJECT_SOURCE_DIR}")
        execute_process(
            COMMAND git -C "${CMAKE_SOURCE_DIR}" worktree add --no-checkout --detach "${LLVM_PROJECT_SOURCE_DIR}" "${LLVM_PROJECT_TAG}"
            COMMAND_ERROR_IS_FATAL ANY
        )
    endif()

    execute_process(
    COMMAND git sparse-checkout init --cone
    WORKING_DIRECTORY "${LLVM_PROJECT_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
    )

    execute_process(
    COMMAND git sparse-checkout set openmp offload cmake llvm/include libc
    WORKING_DIRECTORY "${LLVM_PROJECT_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
    )

    execute_process(
    COMMAND git checkout "${LLVM_PROJECT_TAG}"
    WORKING_DIRECTORY "${LLVM_PROJECT_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
    )

    # Build OpenMP runtime (required dependency for offload's libomptarget) from the cloned source
    ExternalProject_Add(openmp_ext
        # DeviceRTL uses -fuse-ld=lld, so add lld to the dependencies.
        DEPENDS llvm-tblgen LLVMSupport clang lld FileCheck not
        SOURCE_DIR ${LLVM_PROJECT_SOURCE_DIR}/openmp
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${OPENMP_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_BINARY_DIR}/bin/clang
            -DCMAKE_CXX_COMPILER=${CMAKE_BINARY_DIR}/bin/clang++
            -DLIBOMP_OMPD_GDB_SUPPORT=OFF
            -DOPENMP_ENABLE_OMPT_TOOLS=OFF
            -DCMAKE_PREFIX_PATH=${CMAKE_BINARY_DIR}/bin
        INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
        UPDATE_COMMAND ""
        DOWNLOAD_COMMAND ""
    )

    # Build liboffload runtime from the same source tree
    ExternalProject_Add(offload_ext
        DEPENDS openmp_ext
        SOURCE_DIR ${LLVM_PROJECT_SOURCE_DIR}/offload
        LIST_SEPARATOR |
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${UR_OFFLOAD_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_BINARY_DIR}/bin/clang
            -DCMAKE_CXX_COMPILER=${CMAKE_BINARY_DIR}/bin/clang++
            -DLIBOMPTARGET_LLVM_INCLUDE_DIRS=${LLVM_PROJECT_SOURCE_DIR}/llvm/include|${CMAKE_BINARY_DIR}/include
            -DLLVM_DIR=${CMAKE_BINARY_DIR}/lib/cmake/llvm
            -DLIBOMPTARGET_PLUGINS_TO_BUILD=cuda|amdgpu
            -DLIBOMP_INCLUDE_DIR=${OPENMP_INSTALL_DIR}/include
            -DLLVM_TABLEGEN=${CMAKE_BINARY_DIR}/bin/llvm-tblgen
            -DCMAKE_PREFIX_PATH=${OPENMP_INSTALL_DIR}|${CMAKE_BINARY_DIR}/bin
        INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
        UPDATE_COMMAND ""
        DOWNLOAD_COMMAND ""
        BUILD_BYPRODUCTS "${UR_OFFLOAD_INSTALL_DIR}/lib/libLLVMOffload.so"
    )
    add_dependencies(${TARGET_NAME} offload_ext)
    install(DIRECTORY "${UR_OFFLOAD_INSTALL_DIR}/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}"
        COMPONENT ur_adapter_offload)
endif()

# When targetting CUDA devices, we need a workaround to avoid sending PTX to
# liboffload as the CUDA plugin doesn't support it yet. The workaround is to
# simply always link the incoming program so it ends up as CUBIN. Try to find
# the cuda driver so we can enable this where possible.
if (NOT TARGET cudadrv)
    find_package(CUDA 10.1)
    add_library(cudadrv SHARED IMPORTED GLOBAL)
    set_target_properties(
        cudadrv PROPERTIES
        IMPORTED_LOCATION             "${CUDA_cuda_driver_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}"
    )
endif()

install_ur_library(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
)

set(ADDITIONAL_LINK_LIBS "")
if (CUDA_cuda_driver_LIBRARY)
    list(APPEND ADDITIONAL_LINK_LIBS cudadrv)
    target_compile_definitions(${TARGET_NAME} PRIVATE UR_CUDA_ENABLED)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::umf
    ur_common
    ${UR_OFFLOAD_INSTALL_DIR}/lib/libLLVMOffload.so
    ${ADDITIONAL_LINK_LIBS}
)

target_include_directories(${TARGET_NAME} PRIVATE
    "${UR_OFFLOAD_INCLUDE_DIR}/offload"
    "${UR_OFFLOAD_INCLUDE_DIR}/"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../"
)
