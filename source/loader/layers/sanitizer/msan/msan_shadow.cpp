/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_shadow.cpp
 *
 */

#include "msan_shadow.hpp"
#include "msan_interceptor.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "ur_api.h"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {
namespace msan {

#define CPU_SHADOW1_BEGIN 0x010000000000ULL
#define CPU_SHADOW1_END 0x100000000000ULL
#define CPU_SHADOW2_BEGIN 0x200000000000ULL
#define CPU_SHADOW2_END 0x300000000000ULL
#define CPU_SHADOW3_BEGIN 0x500000000000ULL
#define CPU_SHADOW3_END 0x510000000000ULL

#define CPU_SHADOW_MASK 0x500000000000ULL

std::shared_ptr<MsanShadowMemory>
GetMsanShadowMemory(ur_context_handle_t Context, ur_device_handle_t Device,
                    DeviceType Type) {
    if (Type == DeviceType::CPU) {
        static std::shared_ptr<MsanShadowMemory> ShadowCPU =
            std::make_shared<MsanShadowMemoryCPU>(Context, Device);
        return ShadowCPU;
    } else if (Type == DeviceType::GPU_PVC) {
        static std::shared_ptr<MsanShadowMemory> ShadowPVC =
            std::make_shared<MsanShadowMemoryPVC>(Context, Device);
        return ShadowPVC;
    } else if (Type == DeviceType::GPU_DG2) {
        static std::shared_ptr<MsanShadowMemory> ShadowDG2 =
            std::make_shared<MsanShadowMemoryDG2>(Context, Device);
        return ShadowDG2;
    } else {
        getContext()->logger.error("Unsupport device type");
        return nullptr;
    }
}

ur_result_t MsanShadowMemoryCPU::Setup() {
    static ur_result_t Result = [this]() {
        if (MmapFixedNoReserve(CPU_SHADOW1_BEGIN,
                               CPU_SHADOW1_END - CPU_SHADOW1_BEGIN) == 0) {
            return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        if (ProtectMemoryRange(CPU_SHADOW1_END,
                               CPU_SHADOW2_BEGIN - CPU_SHADOW1_END) == 0) {
            return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        if (MmapFixedNoReserve(CPU_SHADOW2_BEGIN,
                               CPU_SHADOW2_END - CPU_SHADOW2_BEGIN) == 0) {
            return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        if (ProtectMemoryRange(CPU_SHADOW2_END,
                               CPU_SHADOW3_BEGIN - CPU_SHADOW2_END) == 0) {
            return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        if (MmapFixedNoReserve(CPU_SHADOW3_BEGIN,
                               CPU_SHADOW3_END - CPU_SHADOW3_BEGIN) == 0) {
            return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
        }
        ShadowBegin = CPU_SHADOW1_BEGIN;
        ShadowEnd = CPU_SHADOW3_END;
        DontCoredumpRange(ShadowBegin, ShadowEnd - ShadowBegin);
        return UR_RESULT_SUCCESS;
    }();
    return Result;
}

ur_result_t MsanShadowMemoryCPU::Destory() {
    if (ShadowBegin == 0 && ShadowEnd == 0) {
        return UR_RESULT_SUCCESS;
    }
    static ur_result_t Result = [this]() {
        if (!Munmap(CPU_SHADOW1_BEGIN, CPU_SHADOW1_END - CPU_SHADOW1_BEGIN)) {
            return UR_RESULT_ERROR_UNKNOWN;
        }
        if (!Munmap(CPU_SHADOW1_END, CPU_SHADOW2_BEGIN - CPU_SHADOW1_END)) {
            return UR_RESULT_ERROR_UNKNOWN;
        }
        if (!Munmap(CPU_SHADOW2_BEGIN, CPU_SHADOW2_END - CPU_SHADOW2_BEGIN) ==
            0) {
            return UR_RESULT_ERROR_UNKNOWN;
        }
        if (!Munmap(CPU_SHADOW2_END, CPU_SHADOW3_BEGIN - CPU_SHADOW2_END)) {
            return UR_RESULT_ERROR_UNKNOWN;
        }
        if (!Munmap(CPU_SHADOW3_BEGIN, CPU_SHADOW3_END - CPU_SHADOW3_BEGIN) ==
            0) {
            return UR_RESULT_ERROR_UNKNOWN;
        }
        ShadowBegin = ShadowEnd = 0;
        return UR_RESULT_SUCCESS;
    }();
    return Result;
}

uptr MsanShadowMemoryCPU::MemToShadow(uptr Ptr) {
    return Ptr ^ CPU_SHADOW_MASK;
}

ur_result_t MsanShadowMemoryCPU::EnqueuePoisonShadow(ur_queue_handle_t,
                                                     uptr Ptr, uptr Size,
                                                     u8 Value) {
    if (Size == 0) {
        return UR_RESULT_SUCCESS;
    }

    uptr ShadowBegin = MemToShadow(Ptr);
    uptr ShadowEnd = MemToShadow(Ptr + Size - 1);
    assert(ShadowBegin <= ShadowEnd);
    getContext()->logger.debug(
        "EnqueuePoisonShadow(addr={}, count={}, value={})", (void *)ShadowBegin,
        ShadowEnd - ShadowBegin + 1, (void *)(size_t)Value);
    memset((void *)ShadowBegin, Value, ShadowEnd - ShadowBegin + 1);

    return UR_RESULT_SUCCESS;
}

ur_result_t MsanShadowMemoryGPU::Setup() {
    // Currently, Level-Zero doesn't create independent VAs for each contexts, if we reserve
    // shadow memory for each contexts, this will cause out-of-resource error when user uses
    // multiple contexts. Therefore, we just create one shadow memory here.
    static ur_result_t Result = [this]() {
        size_t ShadowSize = GetShadowSize();
        // TODO: Protect Bad Zone
        auto Result = getContext()->urDdiTable.VirtualMem.pfnReserve(
            Context, nullptr, ShadowSize, (void **)&ShadowBegin);
        if (Result == UR_RESULT_SUCCESS) {
            ShadowEnd = ShadowBegin + ShadowSize;
            // Retain the context which reserves shadow memory
            getContext()->urDdiTable.Context.pfnRetain(Context);
        }

        // Set shadow memory for null pointer
        ManagedQueue Queue(Context, Device);
        return UR_RESULT_SUCCESS;
    }();
    return Result;
}

ur_result_t MsanShadowMemoryGPU::Destory() {
    if (ShadowBegin == 0) {
        return UR_RESULT_SUCCESS;
    }
    static ur_result_t Result = [this]() {
        auto Result = getContext()->urDdiTable.VirtualMem.pfnFree(
            Context, (const void *)ShadowBegin, GetShadowSize());
        getContext()->urDdiTable.Context.pfnRelease(Context);
        return Result;
    }();
    return Result;
}

ur_result_t MsanShadowMemoryGPU::EnqueuePoisonShadow(ur_queue_handle_t Queue,
                                                     uptr Ptr, uptr Size,
                                                     u8 Value) {
    if (Size == 0) {
        return UR_RESULT_SUCCESS;
    }

    uptr ShadowBegin = MemToShadow(Ptr);
    uptr ShadowEnd = MemToShadow(Ptr + Size - 1);
    assert(ShadowBegin <= ShadowEnd);
    {
        static const size_t PageSize =
            GetVirtualMemGranularity(Context, Device);

        ur_physical_mem_properties_t Desc{
            UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES, nullptr, 0};

        // Make sure [Ptr, Ptr + Size] is mapped to physical memory
        for (auto MappedPtr = RoundDownTo(ShadowBegin, PageSize);
             MappedPtr <= ShadowEnd; MappedPtr += PageSize) {
            std::scoped_lock<ur_mutex> Guard(VirtualMemMapsMutex);
            if (VirtualMemMaps.find(MappedPtr) == VirtualMemMaps.end()) {
                ur_physical_mem_handle_t PhysicalMem{};
                auto URes = getContext()->urDdiTable.PhysicalMem.pfnCreate(
                    Context, Device, PageSize, &Desc, &PhysicalMem);
                if (URes != UR_RESULT_SUCCESS) {
                    getContext()->logger.error("urPhysicalMemCreate(): {}",
                                               URes);
                    return URes;
                }

                URes = getContext()->urDdiTable.VirtualMem.pfnMap(
                    Context, (void *)MappedPtr, PageSize, PhysicalMem, 0,
                    UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE);
                if (URes != UR_RESULT_SUCCESS) {
                    getContext()->logger.error("urVirtualMemMap({}, {}): {}",
                                               (void *)MappedPtr, PageSize,
                                               URes);
                    return URes;
                }

                getContext()->logger.debug("urVirtualMemMap: {} ~ {}",
                                           (void *)MappedPtr,
                                           (void *)(MappedPtr + PageSize - 1));

                // Initialize to zero
                URes = EnqueueUSMBlockingSet(Queue, (void *)MappedPtr, 0,
                                             PageSize);
                if (URes != UR_RESULT_SUCCESS) {
                    getContext()->logger.error("EnqueueUSMBlockingSet(): {}",
                                               URes);
                    return URes;
                }

                VirtualMemMaps[MappedPtr].first = PhysicalMem;
            }

            // We don't need to record virtual memory map for null pointer,
            // since it doesn't have an alloc info.
            if (Ptr == 0) {
                continue;
            }

            auto AllocInfoIt =
                getMsanInterceptor()->findAllocInfoByAddress(Ptr);
            assert(AllocInfoIt);
            VirtualMemMaps[MappedPtr].second.insert((*AllocInfoIt)->second);
        }
    }

    auto URes = EnqueueUSMBlockingSet(Queue, (void *)ShadowBegin, Value,
                                      ShadowEnd - ShadowBegin + 1);
    getContext()->logger.debug(
        "EnqueuePoisonShadow (addr={}, count={}, value={}): {}",
        (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1, (void *)(size_t)Value,
        URes);
    if (URes != UR_RESULT_SUCCESS) {
        getContext()->logger.error("EnqueueUSMBlockingSet(): {}", URes);
        return URes;
    }

    return UR_RESULT_SUCCESS;
}

ur_result_t
MsanShadowMemoryGPU::ReleaseShadow(std::shared_ptr<MsanAllocInfo> AI) {
    uptr ShadowBegin = MemToShadow(AI->AllocBegin);
    uptr ShadowEnd = MemToShadow(AI->AllocBegin + AI->AllocSize);
    assert(ShadowBegin <= ShadowEnd);

    static const size_t PageSize = GetVirtualMemGranularity(Context, Device);

    for (auto MappedPtr = RoundDownTo(ShadowBegin, PageSize);
         MappedPtr <= ShadowEnd; MappedPtr += PageSize) {
        std::scoped_lock<ur_mutex> Guard(VirtualMemMapsMutex);
        if (VirtualMemMaps.find(MappedPtr) == VirtualMemMaps.end()) {
            continue;
        }
        VirtualMemMaps[MappedPtr].second.erase(AI);
        if (VirtualMemMaps[MappedPtr].second.empty()) {
            UR_CALL(getContext()->urDdiTable.VirtualMem.pfnUnmap(
                Context, (void *)MappedPtr, PageSize));
            UR_CALL(getContext()->urDdiTable.PhysicalMem.pfnRelease(
                VirtualMemMaps[MappedPtr].first));
            getContext()->logger.debug("urVirtualMemUnmap: {} ~ {}",
                                       (void *)MappedPtr,
                                       (void *)(MappedPtr + PageSize - 1));
        }
    }

    return UR_RESULT_SUCCESS;
}

uptr MsanShadowMemoryPVC::MemToShadow(uptr Ptr) {
    assert(Ptr & 0xFF00000000000000ULL && "Ptr must be device USM");
    return ShadowBegin + (Ptr & 0x3FFF'FFFF'FFFFULL);
}

uptr MsanShadowMemoryDG2::MemToShadow(uptr Ptr) {
    assert(Ptr & 0xFFFF000000000000ULL && "Ptr must be device USM");
    return ShadowBegin + (Ptr & 0x3FFF'FFFF'FFFFULL);
}

} // namespace msan
} // namespace ur_sanitizer_layer
