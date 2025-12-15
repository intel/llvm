# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_devices_test(exp_graph
  urEnqueueGraphExp.cpp
  urGraphCreateExp.cpp
  urGraphDestroyExp.cpp
  urGraphInstantiateGraphExp.cpp
  urGraphIsEmptyExp.cpp
  urQueueBeginCaptureIntoGraphExp.cpp
  urQueueBeginGraphCaptureExp.cpp
  urQueueEndGraphCaptureExp.cpp
  urQueueIsGraphCaptureEnabledExp.cpp)
