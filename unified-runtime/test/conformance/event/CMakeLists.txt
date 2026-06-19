# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_devices_test(event
    urEventGetInfo.cpp
    urEventGetProfilingInfo.cpp
    urEventWait.cpp
    urEventRetain.cpp
    urEventRelease.cpp
    urEventGetNativeHandle.cpp
    urEventCreateWithNativeHandle.cpp
    urEventSetCallback.cpp)

# The IPC event tests signal the producer event via the native Level Zero API,
# so they are only built and linked when a Level Zero adapter is present.
if(UR_BUILD_ADAPTER_L0 OR UR_BUILD_ADAPTER_L0_V2)
    target_sources(event-test PRIVATE
        urIPCEventExp.cpp
        urIPCEventExpNegative.cpp)
    target_link_libraries(event-test PRIVATE
        LevelZeroLoader LevelZeroLoader-Headers)
endif()
