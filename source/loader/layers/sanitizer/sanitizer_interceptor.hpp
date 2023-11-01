//==---------- sanitizer_interceptor.hpp - Sanitizer interceptor -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ur_ddi.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

typedef uintptr_t uptr;
typedef unsigned char u8;
typedef unsigned int u32;

enum USMMemoryType {
    DEVICE,
    SHARE,
    HOST,
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

  private:
    struct AllocatedMemoryInfo {
        uptr AllocBegin;
        uptr UserBegin;
        uptr UserEnd;
        size_t AllocSize;
        USMMemoryType Type;
        std::unordered_set<ur_device_handle_t> Devices;
    };

    struct DeviceInfo {
        bool Init = false;
        bool InitPool = false;

        ur_device_type_t Type;
        size_t Alignment;
        uptr ShadowOffset;
        uptr ShadowOffsetEnd;

        std::vector<AllocatedMemoryInfo> AllocInfos;
        std::mutex Mutex; // Lock Init & InitPool & AllocInfos
    };

    enum class DeviceType { CPU, GPU_PVC, GPU_DG2 };

    struct ContextInfo {
        bool Init = false;

        /// AllocatedMemoryInfo.AllocBegin => AllocatedMemoryInfo
        ///
        /// Use AllocBegin as key can help to detect underflow pointer
        std::map<uptr, AllocatedMemoryInfo> AllocatedAddressesMap;

        std::vector<AllocatedMemoryInfo> AllocHostInfos;

        /// Each context is able to contain multiple devices
        std::unordered_map<ur_device_handle_t, DeviceInfo> DeviceMap;

        std::mutex Mutex; // Lock Init and Maps

        uptr HostShadowOffset;

        DeviceInfo &getDeviceInfo(ur_device_handle_t Device) {
            std::lock_guard<std::mutex> Guard(Mutex);
            return DeviceMap[Device];
        }
        AllocatedMemoryInfo &getAllocatedMemoryInfo(uptr Address) {
            std::lock_guard<std::mutex> Guard(Mutex);
            return AllocatedAddressesMap[Address];
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
                          AllocatedMemoryInfo &AllocInfo,
                          ur_event_handle_t &LastEvent);
    bool updateHostShadowMemory(ur_context_handle_t Context,
                                AllocatedMemoryInfo AllocInfo);
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
