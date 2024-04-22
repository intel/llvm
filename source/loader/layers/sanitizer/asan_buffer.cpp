/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_buffer.cpp
 *
 */

#include "asan_buffer.hpp"
#include "asan_interceptor.hpp"
#include "ur_sanitizer_layer.hpp"
#include "ur_sanitizer_utils.hpp"

namespace ur_sanitizer_layer {

ur_result_t EnqueueMemCopyRectHelper(
    ur_queue_handle_t Queue, char *pSrc, char *pDst, ur_rect_offset_t SrcOffset,
    ur_rect_offset_t DstOffset, ur_rect_region_t Region, size_t SrcRowPitch,
    size_t SrcSlicePitch, size_t DstRowPitch, size_t DstSlicePitch,
    bool Blocking, uint32_t NumEventsInWaitList,
    const ur_event_handle_t *EventWaitList, ur_event_handle_t *Event) {
    // If user doesn't determine src/dst row pitch and slice pitch, just use
    // region for it.
    if (SrcRowPitch == 0) {
        SrcRowPitch = Region.width;
    }

    if (SrcSlicePitch == 0) {
        SrcSlicePitch = SrcRowPitch * Region.height;
    }

    if (DstRowPitch == 0) {
        DstRowPitch = Region.width;
    }

    if (DstSlicePitch == 0) {
        DstSlicePitch = DstRowPitch * Region.height;
    }

    // Calculate the src and dst addresses that actually will be copied.
    char *SrcOrigin = pSrc + SrcOffset.x + SrcRowPitch * SrcOffset.y +
                      SrcSlicePitch * SrcOffset.z;
    char *DstOrigin = pDst + DstOffset.x + DstRowPitch * DstOffset.y +
                      DstSlicePitch * DstOffset.z;

    std::vector<ur_event_handle_t> Events;
    Events.reserve(Region.depth);
    // For now, USM doesn't support 3D memory copy operation, so we can only
    // loop call 2D memory copy function to implement it.
    for (size_t i = 0; i < Region.depth; i++) {
        ur_event_handle_t NewEvent{};
        UR_CALL(context.urDdiTable.Enqueue.pfnUSMMemcpy2D(
            Queue, Blocking, DstOrigin + (i * DstSlicePitch), DstRowPitch,
            SrcOrigin + (i * SrcSlicePitch), SrcRowPitch, Region.width,
            Region.height, NumEventsInWaitList, EventWaitList, &NewEvent));

        Events.push_back(NewEvent);
    }

    UR_CALL(context.urDdiTable.Enqueue.pfnEventsWait(Queue, Events.size(),
                                                     Events.data(), Event));

    return UR_RESULT_SUCCESS;
}

ur_result_t MemBuffer::getHandle(ur_device_handle_t Device, char *&Handle) {
    // Sub-buffers don't maintain own allocations but rely on parent buffer.
    if (SubBuffer) {
        UR_CALL(SubBuffer->Parent->getHandle(Device, Handle));
        Handle += SubBuffer->Origin;
        return UR_RESULT_SUCCESS;
    }

    auto &Allocation = Allocations[Device];
    if (!Allocation) {
        ur_usm_desc_t USMDesc{};
        USMDesc.align = getAlignment();
        ur_usm_pool_handle_t Pool{};
        ur_result_t URes = context.interceptor->allocateMemory(
            Context, Device, &USMDesc, Pool, Size, AllocType::MEM_BUFFER,
            ur_cast<void **>(&Allocation));
        if (URes != UR_RESULT_SUCCESS) {
            context.logger.error(
                "Failed to allocate {} bytes memory for buffer {}", Size, this);
            return URes;
        }

        if (HostPtr) {
            ManagedQueue Queue(Context, Device);
            URes = context.urDdiTable.Enqueue.pfnUSMMemcpy(
                Queue, true, Allocation, HostPtr, Size, 0, nullptr, nullptr);
            if (URes != UR_RESULT_SUCCESS) {
                context.logger.error("Failed to copy {} bytes data from host "
                                     "pointer {} to buffer {}",
                                     Size, HostPtr, this);
                return URes;
            }
        }
    }

    Handle = Allocation;

    return UR_RESULT_SUCCESS;
}

ur_result_t MemBuffer::free() {
    for (const auto &[_, Ptr] : Allocations) {
        ur_result_t URes = context.interceptor->releaseMemory(Context, Ptr);
        if (URes != UR_RESULT_SUCCESS) {
            context.logger.error("Failed to free buffer handle {}", Ptr);
            return URes;
        }
    }
    Allocations.clear();
    return UR_RESULT_SUCCESS;
}

size_t MemBuffer::getAlignment() {
    // Choose an alignment that is at most 128 and is the next power of 2
    // for sizes less than 128.
    // TODO: If we don't set the alignment size explicitly, the device will
    // usually choose a very large size (more than 1k). Then sanitizer will
    // allocate extra unnessary memory. Not sure if this will impact
    // performance.
    size_t MsbIdx = 63 - __builtin_clz(Size);
    size_t Alignment = (1 << (MsbIdx + 1));
    if (Alignment > 128) {
        Alignment = 128;
    }
    return Alignment;
}

} // namespace ur_sanitizer_layer
