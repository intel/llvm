# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT

add_conformance_test_with_platform_environment(device
    urDeviceCreateWithNativeHandle.cpp
    urDeviceGet.cpp
    urDeviceGetGlobalTimestamps.cpp
    urDeviceGetInfo.cpp
    urDeviceGetNativeHandle.cpp
    urDevicePartition.cpp
    urDeviceRelease.cpp
    urDeviceRetain.cpp
    urDeviceSelectBinary.cpp)
