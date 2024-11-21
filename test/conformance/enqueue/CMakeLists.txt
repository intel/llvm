# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_test_with_kernels_environment(enqueue
    urEnqueueDeviceGlobalVariableRead.cpp
    urEnqueueDeviceGlobalVariableWrite.cpp
    urEnqueueEventsWait.cpp
    urEnqueueEventsWaitMultiDevice.cpp
    urEnqueueEventsWaitWithBarrier.cpp
    urEnqueueKernelLaunch.cpp
    #urEnqueueKernelLaunchAndMemcpyInOrder.cpp
    urEnqueueMemBufferCopyRect.cpp
    urEnqueueMemBufferCopy.cpp
    urEnqueueMemBufferFill.cpp
    urEnqueueMemBufferMap.cpp
    urEnqueueMemBufferRead.cpp
    urEnqueueMemBufferReadRect.cpp
    urEnqueueMemBufferWrite.cpp
    urEnqueueMemBufferWriteRect.cpp
    urEnqueueMemImageCopy.cpp
    urEnqueueMemImageRead.cpp
    urEnqueueMemImageWrite.cpp
    urEnqueueMemUnmap.cpp
    urEnqueueUSMFill.cpp
    urEnqueueUSMFill2D.cpp
    urEnqueueUSMAdvise.cpp
    urEnqueueUSMMemcpy.cpp
    urEnqueueUSMMemcpy2D.cpp
    urEnqueueUSMPrefetch.cpp
    urEnqueueReadHostPipe.cpp
    urEnqueueWriteHostPipe.cpp
    urEnqueueTimestampRecording.cpp
    )
