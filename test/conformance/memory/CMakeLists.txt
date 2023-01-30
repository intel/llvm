# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_conformance_test_with_devices_environment(memory
    urMemBufferCreate.cpp 
    urMemBufferPartition.cpp 
    urMemCreateWithNativeHandle.cpp 
    urMemGetInfo.cpp 
    urMemGetNativeHandle.cpp 
    urMemImageCreate.cpp
    urMemImageGetInfo.cpp 
    urMemRelease.cpp 
    urMemRetain.cpp)
