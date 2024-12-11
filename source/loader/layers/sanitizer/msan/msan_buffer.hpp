/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_buffer.hpp
 *
 */

#pragma once

#include <atomic>
#include <memory>
#include <optional>

#include "ur/ur.hpp"

namespace ur_sanitizer_layer {
namespace msan {

struct MemBuffer {
    // Buffer constructor
    MemBuffer(ur_context_handle_t Context, size_t Size, char *HostPtr)
        : Context(Context), Size(Size), HostPtr(HostPtr) {}

    // Sub-buffer constructor
    MemBuffer(std::shared_ptr<MemBuffer> Parent, size_t Origin, size_t Size)
        : Context(Parent->Context), Size(Size), SubBuffer{{Parent, Origin}} {}

    ur_result_t getHandle(ur_device_handle_t Device, char *&Handle);

    ur_result_t free();

    size_t getAlignment();

    std::unordered_map<ur_device_handle_t, char *> Allocations;

    enum AccessMode { UNKNOWN, READ_WRITE, READ_ONLY, WRITE_ONLY };

    struct Mapping {
        size_t Offset;
        size_t Size;
    };

    std::unordered_map<void *, Mapping> Mappings;

    ur_context_handle_t Context;

    struct Device_t {
        ur_device_handle_t hDevice;
        char *MemHandle;
    };
    Device_t LastSyncedDevice{};

    size_t Size;

    char *HostPtr{};

    struct SubBuffer_t {
        std::shared_ptr<MemBuffer> Parent;
        size_t Origin;
    };

    std::optional<SubBuffer_t> SubBuffer;

    std::atomic<int32_t> RefCount = 1;

    ur_shared_mutex Mutex;
};

ur_result_t EnqueueMemCopyRectHelper(
    ur_queue_handle_t Queue, char *pSrc, char *pDst, ur_rect_offset_t SrcOffset,
    ur_rect_offset_t DstOffset, ur_rect_region_t Region, size_t SrcRowPitch,
    size_t SrcSlicePitch, size_t DstRowPitch, size_t DstSlicePitch,
    bool Blocking, uint32_t NumEventsInWaitList,
    const ur_event_handle_t *EventWaitList, ur_event_handle_t *Event);

} // namespace msan
} // namespace ur_sanitizer_layer
