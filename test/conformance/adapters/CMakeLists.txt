# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(UR_BUILD_ADAPTER_CUDA)
    add_subdirectory(cuda)
endif()

if(UR_BUILD_ADAPTER_HIP)
    add_subdirectory(hip)
endif()
