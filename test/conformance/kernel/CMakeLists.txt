# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_conformance_test_with_kernels_environment(kernel
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
    urKernelSetSpecializationConstants.cpp)
