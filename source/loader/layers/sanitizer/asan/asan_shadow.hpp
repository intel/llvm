/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_shadow.hpp
 *
 */

#pragma once

#include "asan/asan_allocator.hpp"
#include "sanitizer_common/sanitizer_libdevice.hpp"

#include <unordered_set>

namespace ur_sanitizer_layer {
namespace asan {

struct ShadowMemory {
    ShadowMemory(ur_context_handle_t Context, ur_device_handle_t Device)
        : Context(Context), Device(Device) {}

    virtual ~ShadowMemory() {}

    virtual ur_result_t Setup() = 0;

    virtual ur_result_t Destory() = 0;

    virtual uptr MemToShadow(uptr Ptr) = 0;

    virtual ur_result_t EnqueuePoisonShadow(ur_queue_handle_t Queue, uptr Ptr,
                                            uptr Size, u8 Value) = 0;

    virtual size_t GetShadowSize() = 0;

    virtual ur_result_t AllocLocalShadow(ur_queue_handle_t Queue,
                                         uint32_t NumWG, uptr &Begin,
                                         uptr &End) = 0;

    virtual ur_result_t AllocPrivateShadow(ur_queue_handle_t Queue,
                                           uint32_t NumWG, uptr &Begin,
                                           uptr &End) = 0;

    ur_context_handle_t Context{};

    ur_device_handle_t Device{};

    uptr ShadowBegin = 0;

    uptr ShadowEnd = 0;
};

struct ShadowMemoryCPU final : public ShadowMemory {
    ShadowMemoryCPU(ur_context_handle_t Context, ur_device_handle_t Device)
        : ShadowMemory(Context, Device) {}

    ur_result_t Setup() override;

    ur_result_t Destory() override;

    uptr MemToShadow(uptr Ptr) override;

    ur_result_t EnqueuePoisonShadow(ur_queue_handle_t Queue, uptr Ptr,
                                    uptr Size, u8 Value) override;

    size_t GetShadowSize() override { return 0x80000000000ULL; }

    ur_result_t AllocLocalShadow(ur_queue_handle_t, uint32_t, uptr &Begin,
                                 uptr &End) override {
        Begin = ShadowBegin;
        End = ShadowEnd;
        return UR_RESULT_SUCCESS;
    }

    ur_result_t AllocPrivateShadow(ur_queue_handle_t, uint32_t, uptr &Begin,
                                   uptr &End) override {
        Begin = ShadowBegin;
        End = ShadowEnd;
        return UR_RESULT_SUCCESS;
    }
};

struct ShadowMemoryGPU : public ShadowMemory {
    ShadowMemoryGPU(ur_context_handle_t Context, ur_device_handle_t Device)
        : ShadowMemory(Context, Device) {}

    ur_result_t Setup() override;

    ur_result_t Destory() override;
    ur_result_t EnqueuePoisonShadow(ur_queue_handle_t Queue, uptr Ptr,
                                    uptr Size, u8 Value) override final;

    ur_result_t AllocLocalShadow(ur_queue_handle_t Queue, uint32_t NumWG,
                                 uptr &Begin, uptr &End) override final;

    ur_result_t AllocPrivateShadow(ur_queue_handle_t Queue, uint32_t NumWG,
                                   uptr &Begin, uptr &End) override final;

    ur_mutex VirtualMemMapsMutex;

    std::unordered_map<uptr, ur_physical_mem_handle_t> VirtualMemMaps;

    uptr LocalShadowOffset = 0;

    uptr PrivateShadowOffset = 0;
};

/// Shadow Memory layout of GPU PVC device
///
/// USM Allocation Range (56 bits)
///   Host   USM : 0x0000_0000_0000_0000 ~ 0x00ff_ffff_ffff_ffff
///   Shared USM : 0x0000_0000_0000_0000 ~ 0x0000_7fff_ffff_ffff
///   Device USM : 0xff00_0000_0000_0000 ~ 0xff00_ffff_ffff_ffff
///
/// USM Allocation Range (AllocateHostAllocationsInHeapExtendedHost=0)
///   Host   USM : 0x0000_0000_0000_0000 ~ 0x0000_7fff_ffff_ffff
///   Shared USM : 0x0000_0000_0000_0000 ~ 0x0000_7fff_ffff_ffff
///   Device USM : 0xff00_0000_0000_0000 ~ 0xff00_ffff_ffff_ffff
///
/// Shadow Memory Mapping (SHADOW_SCALE=4, AllocateHostAllocationsInHeapExtendedHost=0)
///   Host/Shared USM : 0x0              ~ 0x07ff_ffff_ffff
///   Device USM      : 0x0800_0000_0000 ~ 0x17ff_ffff_ffff
///
struct ShadowMemoryPVC final : public ShadowMemoryGPU {
    ShadowMemoryPVC(ur_context_handle_t Context, ur_device_handle_t Device)
        : ShadowMemoryGPU(Context, Device) {}

    uptr MemToShadow(uptr Ptr) override;

    size_t GetShadowSize() override { return 0x180000000000ULL; }
};

/// Shadow Memory layout of GPU PVC device
///
/// USM Allocation Range (48 bits)
///   Host/Shared USM : 0x0000_0000_0000_0000 ~ 0x0000_7fff_ffff_ffff
///   Device      USM : 0xffff_8000_0000_0000 ~ 0xffff_ffff_ffff_ffff
///
/// Shadow Memory Mapping (SHADOW_SCALE=4)
///   Host/Shared USM : 0x0              ~ 0x07ff_ffff_ffff
///   Device      USM : 0x0800_0000_0000 ~ 0x0fff_ffff_ffff
///
struct ShadowMemoryDG2 final : public ShadowMemoryGPU {
    ShadowMemoryDG2(ur_context_handle_t Context, ur_device_handle_t Device)
        : ShadowMemoryGPU(Context, Device) {}

    uptr MemToShadow(uptr Ptr) override;

    size_t GetShadowSize() override { return 0x100000000000ULL; }
};

std::shared_ptr<ShadowMemory> GetShadowMemory(ur_context_handle_t Context,
                                              ur_device_handle_t Device,
                                              DeviceType Type);

} // namespace asan
} // namespace ur_sanitizer_layer
