# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_conformance_test_with_devices_environment(memory
    urMemBufferCreate.cpp 
    urMemBufferCreateWithNativeHandle.cpp 
    urMemBufferPartition.cpp 
    urMemGetInfo.cpp 
    urMemGetNativeHandle.cpp 
    urMemImageCreate.cpp
    urMemImageCreateWithNativeHandle.cpp 
    urMemImageGetInfo.cpp 
    urMemRelease.cpp 
    urMemRetain.cpp)
