# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_kernels_test(program
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
