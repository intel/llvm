# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_devices_test(virtual_memory
    urPhysicalMemCreate.cpp
    urPhysicalMemRelease.cpp
    urPhysicalMemRetain.cpp
    urPhysicalMemGetInfo.cpp
    urVirtualMemFree.cpp
    urVirtualMemGetInfo.cpp
    urVirtualMemGranularityGetInfo.cpp
    urVirtualMemMap.cpp
    urVirtualMemReserve.cpp
    urVirtualMemSetAccess.cpp
    urVirtualMemUnmap.cpp
)
