# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_conformance_test_with_devices_environment(context 
    urContextCreate.cpp
    urContextCreateWithNativeHandle.cpp
    urContextGetInfo.cpp
    urContextGetNativeHandle.cpp
    urContextRelease.cpp
    urContextRetain.cpp
    urContextSetExtendedDeleter.cpp)
