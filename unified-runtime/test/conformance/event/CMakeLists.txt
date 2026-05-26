# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_devices_test(event
    urEventGetInfo.cpp
    urEventGetProfilingInfo.cpp
    urEventWait.cpp
    urEventRetain.cpp
    urEventRelease.cpp
    urEventGetNativeHandle.cpp
    urEventCreateWithNativeHandle.cpp
    urEventSetCallback.cpp)
