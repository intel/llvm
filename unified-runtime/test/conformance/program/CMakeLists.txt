# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_test_with_kernels_environment(program
    urProgramBuild.cpp
    urProgramCompile.cpp
    urProgramCreateWithBinary.cpp
    urMultiDeviceProgramCreateWithBinary.cpp
    urMultiDeviceProgramCreateWithIL.cpp
    urProgramCreateWithIL.cpp
    urProgramCreateWithNativeHandle.cpp
    urProgramGetBuildInfo.cpp
    urProgramGetFunctionPointer.cpp
    urProgramGetGlobalVariablePointer.cpp
    urProgramGetInfo.cpp
    urProgramGetNativeHandle.cpp
    urProgramLink.cpp
    urProgramRelease.cpp
    urProgramRetain.cpp
    urProgramSetSpecializationConstants.cpp)
