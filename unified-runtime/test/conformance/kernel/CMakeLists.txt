# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
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
    urKernelSetExecInfo.cpp
    urKernelSetSpecializationConstants.cpp
    urKernelSuggestMaxCooperativeGroupCount.cpp
    urKernelGetSuggestedLocalWorkSize.cpp)
