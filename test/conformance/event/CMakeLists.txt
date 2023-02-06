# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_conformance_test_with_devices_environment(event
    urEventGetInfo.cpp
    urEventGetProfilingInfo.cpp
    urEventWait.cpp
    urEventRetain.cpp
    urEventRelease.cpp
    urEventGetNativeHandle.cpp
    urEventCreateWithNativeHandle.cpp
    urEventSetCallback.cpp)
