# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_test_with_devices_environment(event
    urEventGetInfo.cpp
    urEventGetProfilingInfo.cpp
    urEventWait.cpp
    urEventRetain.cpp
    urEventRelease.cpp
    urEventGetNativeHandle.cpp
    urEventCreateWithNativeHandle.cpp
    urEventSetCallback.cpp)
