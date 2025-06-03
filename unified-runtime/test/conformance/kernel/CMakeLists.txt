# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_kernels_test(kernel
    urKernelCreate.cpp
    urKernelCreateWithNativeHandle.cpp
    urKernelGetGroupInfo.cpp
    urKernelGetInfo.cpp
    urKernelGetNativeHandle.cpp
    urKernelGetSubGroupInfo.cpp
    urKernelRelease.cpp
    urKernelRetain.cpp
    urKernelSetArgLocal.cpp
    urKernelSetArgMemObj.cpp
    urKernelSetArgPointer.cpp
    urKernelSetArgSampler.cpp
    urKernelSetArgValue.cpp
    urKernelSetExecInfo.cpp
    urKernelSetSpecializationConstants.cpp
    urKernelSuggestMaxCooperativeGroupCount.cpp
    urKernelGetSuggestedLocalWorkSize.cpp)
