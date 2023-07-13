# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(null)


if(UR_BUILD_ADAPTER_L0 OR UR_BUILD_ADAPTER_CUDA)
    # fetch adapter sources from SYCL
    set(SYCL_ADAPTER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")
    FetchSource(https://github.com/intel/llvm.git sycl-nightly/20230706 "sycl/plugins/unified_runtime/ur" ${SYCL_ADAPTER_DIR})

    get_program_version_major_minor(git GIT_VERSION)
    set(GIT_QUIET_OPTION "")
    if(GIT_VERSION VERSION_GREATER_EQUAL "3.35.0")
        set(GIT_QUIET_OPTION "--quiet")
    endif()
    execute_process(COMMAND git apply ${GIT_QUIET_OPTION} ../0001-SYCL-CUDA-remove-sycl-namespaces-from-ur-adapter.patch
        WORKING_DIRECTORY ${SYCL_ADAPTER_DIR})
endif()

if(UR_BUILD_ADAPTER_L0)
add_subdirectory(level_zero)
endif()

if(UR_BUILD_ADAPTER_CUDA)
add_subdirectory(cuda)
endif()
