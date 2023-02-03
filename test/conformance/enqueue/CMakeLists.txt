# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT

add_conformance_test_with_devices_environment(enqueue
    urEnqueueKernelLaunch.cpp
    urEnqueueEventsWait.cpp
    urEnqueueEventsWaitWithBarrier.cpp
    urEnqueueMemBufferRead.cpp
    urEnqueueMemBufferWrite.cpp
    urEnqueueMemBufferReadRect.cpp
    urEnqueueMemBufferWriteRect.cpp
    urEnqueueMemBufferCopy.cpp
    urEnqueueMemBufferCopyRect.cpp
    urEnqueueMemBufferFill.cpp
    urEnqueueMemImageRead.cpp
    urEnqueueMemImageWrite.cpp
    urEnqueueMemImageCopy.cpp
    urEnqueueMemImageFill.cpp
    urEnqueueMemBufferMap.cpp
    urEnqueueMemImageMap.cpp
    urEnqueueMemUnmap.cpp
    urEnqueueUSMMemset.cpp
    urEnqueueUSMMemcpy.cpp
    urEnqueueUSMPrefetch.cpp
    urEnqueueUSMMemAdvice.cpp
    urEnqueueUSMFill2D.cpp
    urEnqueueUSMMemset2D.cpp
    urEnqueueUSMMemcpy2D.cpp)
