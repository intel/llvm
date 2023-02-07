# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_conformance_test_with_devices_environment(usm 
    urUSMHostAlloc.cpp 
    urUSMDeviceAlloc.cpp
    urUSMFree.cpp
    urUSMGetMemAllocInfo.cpp
    urUSMSharedAlloc.cpp)
