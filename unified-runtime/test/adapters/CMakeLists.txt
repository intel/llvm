# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_custom_target(check-unified-runtime-adapter)

if(UR_BUILD_ADAPTER_CUDA OR UR_BUILD_ADAPTER_ALL)
    add_subdirectory(cuda)
endif()

if(UR_BUILD_ADAPTER_HIP OR UR_BUILD_ADAPTER_ALL)
    add_subdirectory(hip)
endif()

if(UR_BUILD_ADAPTER_L0 OR UR_BUILD_ADAPTER_L0_V2 OR UR_BUILD_ADAPTER_ALL)
    add_subdirectory(level_zero)
endif()
