# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_devices_test(memory
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
