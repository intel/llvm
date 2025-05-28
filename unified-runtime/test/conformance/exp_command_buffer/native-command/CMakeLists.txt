# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if (UR_BUILD_ADAPTER_CUDA)
    target_sources(exp_command_buffer-test PRIVATE "append_native_cuda.cpp")
    target_link_libraries(exp_command_buffer-test PRIVATE cudadrv)
endif()

if (UR_BUILD_ADAPTER_HIP)
    target_sources(exp_command_buffer-test PRIVATE "append_native_hip.cpp")
    target_link_libraries(exp_command_buffer-test PRIVATE rocmdrv)
endif()

if (UR_BUILD_ADAPTER_L0 OR UR_BUILD_ADAPTER_L0_v2)
    target_sources(exp_command_buffer-test PRIVATE "append_native_L0.cpp")
    target_link_libraries(exp_command_buffer-test PRIVATE LevelZeroLoader LevelZeroLoader-Headers)
endif()

if (UR_BUILD_ADAPTER_OPENCL)
    target_sources(exp_command_buffer-test PRIVATE "append_native_opencl.cpp")
    target_compile_definitions(exp_command_buffer-test PRIVATE CL_TARGET_OPENCL_VERSION=300)
    target_link_libraries(exp_command_buffer-test PRIVATE OpenCLICDLoader)
endif()
