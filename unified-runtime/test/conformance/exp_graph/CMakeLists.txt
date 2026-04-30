# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_devices_test(exp_graph
  urEnqueueGraphExp.cpp
  urGraphCreateExp.cpp
  urGraphDestroyExp.cpp
  urGraphDumpContentsExp.cpp
  urGraphInstantiateGraphExp.cpp
  urGraphIsEmptyExp.cpp
  urQueueBeginCaptureIntoGraphExp.cpp
  urQueueBeginGraphCaptureExp.cpp
  urQueueEndGraphCaptureExp.cpp
  urQueueIsGraphCaptureEnabledExp.cpp
  urQueueGetGraphExp.cpp)
