# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_conformance_kernels_test(enqueue
    urEnqueueDeviceGlobalVariableRead.cpp
    urEnqueueDeviceGlobalVariableWrite.cpp
    urEnqueueEventsWait.cpp
    urEnqueueEventsWaitMultiDevice.cpp
    urEnqueueEventsWaitWithBarrier.cpp
    urEnqueueKernelLaunchWithArgsExp.cpp
    urEnqueueKernelLaunchAndMemcpyInOrder.cpp
    urEnqueueMemBufferCopyRect.cpp
    urEnqueueMemBufferCopy.cpp
    urEnqueueMemBufferFill.cpp
    urEnqueueMemBufferMap.cpp
    # Enqueue memory buffer read tests ran simultenously lead to timeouts.
    # Tracker: https://jira.devtools.intel.com/browse/URT-1026
    # urEnqueueMemBufferRead.cpp
    # urEnqueueMemBufferReadRect.cpp
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
