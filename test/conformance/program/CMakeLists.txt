# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_conformance_test_with_kernels_environment(program
    urProgramBuild.cpp
    urProgramCompile.cpp
    urProgramCreateWithBinary.cpp
    urProgramCreateWithIL.cpp
    urProgramCreateWithNativeHandle.cpp
    urProgramGetBuildInfo.cpp
    urProgramGetFunctionPointer.cpp
    urProgramGetInfo.cpp
    urProgramGetNativeHandle.cpp
    urProgramLink.cpp
    urProgramRelease.cpp
    urProgramRetain.cpp
    urProgramSetSpecializationConstants.cpp)
