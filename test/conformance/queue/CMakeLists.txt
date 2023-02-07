# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_conformance_test_with_devices_environment(queue 
  urQueueCreate.cpp 
  urQueueCreateWithNativeHandle.cpp 
  urQueueFinish.cpp
  urQueueFlush.cpp
  urQueueGetInfo.cpp
  urQueueGetNativeHandle.cpp 
  urQueueRetain.cpp
  urQueueRelease.cpp)
