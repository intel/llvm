/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_sanitizer_layer.cpp
 *
 */

#pragma once

#include "common.hpp"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ur_sanitizer_layer {

enum USMMemoryType { DEVICE, SHARE, HOST, MEM_BUFFER };

struct USMAllocInfo {
    uptr AllocBegin;
    uptr UserBegin;
    uptr UserEnd;
    size_t AllocSize;
    USMMemoryType Type;
};

enum class DeviceType { UNKNOWN, CPU, GPU_PVC, GPU_DG2 };

struct DeviceInfo {
    DeviceType Type;
    size_t Alignment;
    uptr ShadowOffset;
    uptr ShadowOffsetEnd;

    // Lock InitPool & AllocInfos
    ur_shared_mutex Mutex;
    std::vector<std::shared_ptr<USMAllocInfo>> AllocInfos;
};

struct QueueInfo {
    ur_mutex Mutex;
    ur_event_handle_t LastEvent;
};

struct ContextInfo {

    std::shared_ptr<DeviceInfo> getDeviceInfo(ur_device_handle_t Device) {
        std::shared_lock<ur_shared_mutex> Guard(Mutex);
        assert(DeviceMap.find(Device) != DeviceMap.end());
        return DeviceMap[Device];
    }

    std::shared_ptr<QueueInfo> getQueueInfo(ur_queue_handle_t Queue) {
        std::shared_lock<ur_shared_mutex> Guard(Mutex);
        assert(QueueMap.find(Queue) != QueueMap.end());
        return QueueMap[Queue];
    }

    std::shared_ptr<USMAllocInfo> getUSMAllocInfo(uptr Address) {
        std::shared_lock<ur_shared_mutex> Guard(Mutex);
        assert(AllocatedUSMMap.find(Address) != AllocatedUSMMap.end());
        return AllocatedUSMMap[Address];
    }

    ur_shared_mutex Mutex;
    std::unordered_map<ur_device_handle_t, std::shared_ptr<DeviceInfo>>
        DeviceMap;
    std::unordered_map<ur_queue_handle_t, std::shared_ptr<QueueInfo>> QueueMap;

    /// key: USMAllocInfo.AllocBegin
    /// value: USMAllocInfo
    /// Use AllocBegin as key can help to detect underflow pointer
    std::map<uptr, std::shared_ptr<USMAllocInfo>> AllocatedUSMMap;
};

class SanitizerInterceptor {
  public:
    ur_result_t allocateMemory(ur_context_handle_t Context,
                               ur_device_handle_t Device,
                               const ur_usm_desc_t *Properties,
                               ur_usm_pool_handle_t Pool, size_t Size,
                               void **ResultPtr, USMMemoryType Type);
    ur_result_t releaseMemory(ur_context_handle_t Context, void *Ptr);

    bool preLaunchKernel(ur_kernel_handle_t Kernel, ur_queue_handle_t Queue,
                         ur_event_handle_t &Event);
    void postLaunchKernel(ur_kernel_handle_t Kernel, ur_queue_handle_t Queue,
                          ur_event_handle_t &Event);

    ur_result_t insertContext(ur_context_handle_t Context);
    ur_result_t eraseContext(ur_context_handle_t Context);

    ur_result_t insertDevice(ur_context_handle_t Context,
                             ur_device_handle_t Device);

    ur_result_t insertQueue(ur_context_handle_t Context,
                            ur_queue_handle_t Queue);
    ur_result_t eraseQueue(ur_context_handle_t Context,
                           ur_queue_handle_t Queue);

  private:
    ur_result_t updateShadowMemory(ur_queue_handle_t Queue);
    ur_result_t enqueueAllocInfo(ur_context_handle_t Context,
                                 ur_device_handle_t Device,
                                 ur_queue_handle_t Queue,
                                 std::shared_ptr<USMAllocInfo> &AlloccInfo,
                                 ur_event_handle_t &LastEvent);

    /// Initialize Global Variables & Kernel Name at first Launch
    void prepareLaunch(ur_queue_handle_t Queue, ur_kernel_handle_t Kernel);

    std::string getKernelName(ur_kernel_handle_t Kernel);
    ur_result_t allocShadowMemory(ur_context_handle_t Context,
                                  std::shared_ptr<DeviceInfo> &DeviceInfo);
    ur_result_t enqueueMemSetShadow(ur_context_handle_t Context,
                                    ur_device_handle_t Device,
                                    ur_queue_handle_t Queue, uptr Addr,
                                    uptr Size, u8 Value,
                                    ur_event_handle_t DepEvent,
                                    ur_event_handle_t *OutEvent);

    std::shared_ptr<ContextInfo> getContextInfo(ur_context_handle_t Context) {
        std::shared_lock<ur_shared_mutex> Guard(m_ContextMapMutex);
        assert(m_ContextMap.find(Context) != m_ContextMap.end());
        return m_ContextMap[Context];
    }

  private:
    std::unordered_map<ur_context_handle_t, std::shared_ptr<ContextInfo>>
        m_ContextMap;
    ur_shared_mutex m_ContextMapMutex;
};

} // namespace ur_sanitizer_layer
