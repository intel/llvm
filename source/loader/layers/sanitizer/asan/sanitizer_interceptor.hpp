//==---------- sanitizer_interceptor.hpp - Sanitizer interceptor -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.h"

#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ur_sanitizer_layer {

enum USMMemoryType {
    DEVICE,
    SHARE,
    HOST,
    MEM_BUFFER
};

class SanitizerInterceptor {
  public:
    SanitizerInterceptor(ur_dditable_t &dditable) : m_Dditable(dditable) {}

    ur_result_t allocateMemory(ur_context_handle_t Context,
                               ur_device_handle_t Device,
                               const ur_usm_desc_t *Properties,
                               ur_usm_pool_handle_t Pool, size_t Size,
                               void **ResultPtr, USMMemoryType Type);
    ur_result_t releaseMemory(ur_context_handle_t Context, void *Ptr);
    void addQueue(ur_context_handle_t Context, ur_device_handle_t Device,
                  ur_queue_handle_t Queue);
    void addKernel(ur_program_handle_t Program, ur_kernel_handle_t Kernel);
    bool launchKernel(ur_kernel_handle_t Kernel, ur_queue_handle_t Queue,
                      ur_event_handle_t &Event);
    void postLaunchKernel(ur_kernel_handle_t Kernel, ur_queue_handle_t Queue,
                          ur_event_handle_t *Event, bool SetCallback = true);
    void checkSanitizerError(ur_kernel_handle_t Kernel);
    ur_result_t createMemoryBuffer(ur_context_handle_t Context,
                                   ur_mem_flags_t Flags, size_t Size,
                                   const ur_buffer_properties_t *Properties,
                                   ur_mem_handle_t *Buffer);

  private:
    struct USMMemoryInfo {
        uptr AllocBegin;
        uptr UserBegin;
        uptr UserEnd;
        size_t AllocSize;
        USMMemoryType Type;
        std::unordered_set<ur_device_handle_t>
            Devices; // host USM have a lot of devices
    };

    struct DeviceInfo {
        bool Init = false;
        bool InitPool = false;

        ur_device_type_t Type;
        size_t Alignment;
        uptr ShadowOffset;
        uptr ShadowOffsetEnd;

        std::vector<USMMemoryInfo> AllocInfos;
        std::mutex Mutex; // Lock Init & InitPool & AllocInfos
    };

    struct MemBufferInfo {
        ur_mem_handle_t Buffer;
        USMMemoryInfo AllocInfo;
        size_t Size;
        size_t RZSize;
    };

    enum class DeviceType { CPU, GPU_PVC, GPU_DG2 };

    struct ContextInfo {
        bool Init = false;

        /// USMMemoryInfo.AllocBegin => USMMemoryInfo
        ///
        /// Use AllocBegin as key can help to detect underflow pointer
        std::map<uptr, USMMemoryInfo> AllocatedAddressesMap;

        std::vector<USMMemoryInfo> AllocHostInfos;

        /// Each context is able to contain multiple devices
        std::unordered_map<ur_device_handle_t, DeviceInfo> DeviceMap;

        ///
        std::unordered_map<ur_mem_handle_t, MemBufferInfo> MemBufferMap;

        std::mutex Mutex; // Lock Init and Maps

        uptr HostShadowOffset;

        DeviceInfo &getDeviceInfo(ur_device_handle_t Device) {
            std::lock_guard<std::mutex> Guard(Mutex);
            return DeviceMap[Device];
        }
        USMMemoryInfo &getUSMMemoryInfo(uptr Address) {
            std::lock_guard<std::mutex> Guard(Mutex);
            return AllocatedAddressesMap[Address];
        }
        MemBufferInfo &getMemBufferInfo(ur_mem_handle_t MemBuffer) {
            std::lock_guard<std::mutex> Guard(Mutex);
            return MemBufferMap[MemBuffer];
        }
    };

    struct QueueInfo {
        ur_device_handle_t Device;
        ur_context_handle_t Context;
        ur_event_handle_t LastEvent = nullptr;
        std::mutex Mutex; // Lock LastEvent
    };

    struct KernelInfo {
        ur_program_handle_t Program = nullptr;
        std::string Name;
        std::mutex Mutex; // Lock Name
    };

  private:
    bool updateShadowMemory(ur_queue_handle_t Queue, ur_kernel_handle_t Kernel);
    void enqueueAllocInfo(ur_context_handle_t Context,
                          ur_device_handle_t Device, ur_queue_handle_t Queue,
                          USMMemoryInfo &AllocInfo,
                          ur_event_handle_t &LastEvent);
    bool updateHostShadowMemory(ur_context_handle_t Context,
                                USMMemoryInfo AllocInfo);
    /// Initialize Global Variables & Kernel Name at first Launch
    bool initKernel(ur_queue_handle_t Queue, ur_kernel_handle_t Kernel);
    /// Initialze USM Host Memory Pools
    void initContext(ur_context_handle_t Context);
    /// Initialze USM Device & Shared Memory Pools, Privte & Local Memory Shadow
    /// Pools
    void initDevice(ur_context_handle_t Context, ur_device_handle_t Device);
    std::string getKernelName(ur_kernel_handle_t Kernel);
    ur_result_t piextMemAllocShadow(ur_context_handle_t Context,
                                    ur_device_handle_t Device);
    ur_result_t piextEnqueueMemSetShadow(ur_context_handle_t Context,
                                         ur_device_handle_t Device,
                                         ur_queue_handle_t Queue, void *Addr,
                                         size_t Size, uint8_t Value,
                                         size_t NumEvents,
                                         const ur_event_handle_t *EventsList,
                                         ur_event_handle_t *OutEvent);
    ur_result_t enqueuePoisonShadow(ur_context_handle_t Context,
                                    ur_device_handle_t Device,
                                    ur_queue_handle_t Queue, uptr Addr,
                                    uptr Size, u8 Value,
                                    ur_event_handle_t DepEvent,
                                    ur_event_handle_t *OutEvent);

  private:
    std::unordered_map<ur_context_handle_t, ContextInfo> m_ContextMap;
    std::mutex m_ContextMapMutex;
    std::unordered_map<ur_queue_handle_t, QueueInfo> m_QueueMap;
    std::mutex m_QueueMapMutex;
    std::unordered_map<ur_kernel_handle_t, KernelInfo> m_KernelMap;
    std::mutex m_KernelMapMutex;

    ContextInfo &getContextInfo(ur_context_handle_t Context) {
        std::lock_guard<std::mutex> ContextMapGuard(m_ContextMapMutex);
        return m_ContextMap[Context];
    }
    QueueInfo &getQueueInfo(ur_queue_handle_t Queue) {
        std::lock_guard<std::mutex> QueueMapGuard(m_QueueMapMutex);
        return m_QueueMap[Queue];
    }
    KernelInfo &getKernelInfo(ur_kernel_handle_t Kernel) {
        std::lock_guard<std::mutex> KernelMapGuard(m_KernelMapMutex);
        return m_KernelMap[Kernel];
    }

    ur_dditable_t &m_Dditable;
};

} // namespace ur_sanitizer_layer
