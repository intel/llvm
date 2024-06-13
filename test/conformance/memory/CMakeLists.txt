# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_test_with_devices_environment(memory
    urMemBufferCreate.cpp 
    urMemBufferCreateWithNativeHandle.cpp 
    urMemBufferPartition.cpp 
    urMemGetInfo.cpp 
    urMemGetNativeHandle.cpp 
    urMemImageCreate.cpp
    urMemImageCreateWithImageFormatParam.cpp
    urMemImageCreateWithNativeHandle.cpp 
    urMemImageGetInfo.cpp 
    urMemRelease.cpp 
    urMemRetain.cpp)

if (UR_DPCXX)
    add_conformance_test_with_kernels_environment(memory-migrate
        urMemBufferMigrateAcrossDevices.cpp)
endif()
