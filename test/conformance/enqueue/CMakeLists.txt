# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

add_conformance_test_with_devices_environment(enqueue
    urEnqueueEventsWait.cpp
    urEnqueueEventsWaitWithBarrier.cpp
    urEnqueueKernelLaunch.cpp
    urEnqueueMemBufferCopyRect.cpp
    urEnqueueMemBufferCopy.cpp
    urEnqueueMemBufferFill.cpp
    urEnqueueMemBufferMap.cpp
    urEnqueueMemBufferRead.cpp
    urEnqueueMemBufferReadRect.cpp
    urEnqueueMemBufferWrite.cpp
    urEnqueueMemBufferWriteRect.cpp
    urEnqueueMemImageCopy.cpp
    urEnqueueMemImageFill.cpp
    urEnqueueMemImageMap.cpp
    urEnqueueMemImageRead.cpp
    urEnqueueMemImageWrite.cpp
    urEnqueueMemUnmap.cpp
    urEnqueueUSMFill.cpp
    urEnqueueUSMFill2D.cpp
    urEnqueueUSMAdvise.cpp
    urEnqueueUSMMemcpy.cpp
    urEnqueueUSMMemcpy2D.cpp
    urEnqueueUSMPrefetch.cpp
    )
