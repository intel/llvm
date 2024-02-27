/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_interceptor.hpp
 *
 */

#pragma once

#include "asan_allocator.hpp"
#include "common.hpp"
#include "device_sanitizer_report.hpp"
#include "ur_sanitizer_layer.hpp"

#include <memory>
#include <queue>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ur_sanitizer_layer {

class Quarantine;

struct USMAllocInfoList {
    std::vector<std::shared_ptr<USMAllocInfo>> List;
    ur_shared_mutex Mutex;
};

struct DeviceInfo {
    ur_device_handle_t Handle;

    DeviceType Type = DeviceType::UNKNOWN;
    size_t Alignment = 0;
    uptr ShadowOffset = 0;
    uptr ShadowOffsetEnd = 0;

    ur_mutex Mutex;
    std::queue<std::shared_ptr<USMAllocInfo>> Quarantine;
    size_t QuarantineSize = 0;

    DeviceInfo(ur_device_handle_t Device) : Handle(Device) {
        [[maybe_unused]] auto Result =
            context.urDdiTable.Device.pfnRetain(Device);
        assert(Result == UR_RESULT_SUCCESS);
    }

    ~DeviceInfo() {
        [[maybe_unused]] auto Result =
            context.urDdiTable.Device.pfnRelease(Handle);
        assert(Result == UR_RESULT_SUCCESS);
    }

    ur_result_t allocShadowMemory(ur_context_handle_t Context);
};

struct QueueInfo {
    ur_queue_handle_t Handle;
    ur_mutex Mutex;
    ur_event_handle_t LastEvent;

    QueueInfo(ur_queue_handle_t Queue) : Handle(Queue), LastEvent(nullptr) {
        [[maybe_unused]] auto Result =
            context.urDdiTable.Queue.pfnRetain(Queue);
        assert(Result == UR_RESULT_SUCCESS);
    }

    ~QueueInfo() {
        [[maybe_unused]] auto Result =
            context.urDdiTable.Queue.pfnRelease(Handle);
        assert(Result == UR_RESULT_SUCCESS);
    }

    static std::unique_ptr<QueueInfo> Create(ur_context_handle_t Context,
                                             ur_device_handle_t Device) {
        ur_queue_handle_t Queue{};
        [[maybe_unused]] auto Result = context.urDdiTable.Queue.pfnCreate(
            Context, Device, nullptr, &Queue);
        assert(Result == UR_RESULT_SUCCESS);
        return std::make_unique<QueueInfo>(Queue);
    }
};

struct ContextInfo {
    ur_context_handle_t Handle;

    std::vector<ur_device_handle_t> DeviceList;
    std::unordered_map<ur_device_handle_t, USMAllocInfoList> AllocInfosMap;

    ContextInfo(ur_context_handle_t Context) : Handle(Context) {
        [[maybe_unused]] auto Result =
            context.urDdiTable.Context.pfnRetain(Context);
        assert(Result == UR_RESULT_SUCCESS);
    }

    ~ContextInfo() {
        [[maybe_unused]] auto Result =
            context.urDdiTable.Context.pfnRelease(Handle);
        assert(Result == UR_RESULT_SUCCESS);
    }

    void insertAllocInfo(const std::vector<ur_device_handle_t> &Devices,
                         std::shared_ptr<USMAllocInfo> &AllocInfo) {
        for (auto Device : Devices) {
            auto &AllocInfos = AllocInfosMap[Device];
            std::scoped_lock<ur_shared_mutex> Guard(AllocInfos.Mutex);
            AllocInfos.List.emplace_back(AllocInfo);
        }
    }
};

struct LaunchInfo {
    uptr LocalShadowOffset;
    uptr LocalShadowOffsetEnd;
    ur_context_handle_t Context;

    DeviceSanitizerReport SPIR_DeviceSanitizerReportMem;

    size_t LocalWorkSize[3];

    LaunchInfo()
        : LocalShadowOffset(0), LocalShadowOffsetEnd(0), Context(nullptr) {}
    ~LaunchInfo();
};

class SanitizerInterceptor {
  public:
    explicit SanitizerInterceptor();

    ~SanitizerInterceptor();

    ur_result_t allocateMemory(ur_context_handle_t Context,
                               ur_device_handle_t Device,
                               const ur_usm_desc_t *Properties,
                               ur_usm_pool_handle_t Pool, size_t Size,
                               void **ResultPtr, MemoryType Type);
    ur_result_t releaseMemory(ur_context_handle_t Context, void *Ptr);

    ur_result_t preLaunchKernel(ur_kernel_handle_t Kernel,
                                ur_queue_handle_t Queue, LaunchInfo &LaunchInfo,
                                uint32_t numWorkgroup);
    void postLaunchKernel(ur_kernel_handle_t Kernel, ur_queue_handle_t Queue,
                          ur_event_handle_t &Event, LaunchInfo &LaunchInfo);

    ur_result_t insertContext(ur_context_handle_t Context,
                              std::shared_ptr<ContextInfo> &CI);
    ur_result_t eraseContext(ur_context_handle_t Context);

    ur_result_t insertDevice(ur_device_handle_t Device,
                             std::shared_ptr<DeviceInfo> &CI);
    ur_result_t eraseDevice(ur_device_handle_t Device);

    ur_result_t insertQueue(ur_context_handle_t Context,
                            ur_queue_handle_t Queue);
    ur_result_t eraseQueue(ur_context_handle_t Context,
                           ur_queue_handle_t Queue);

    std::vector<std::shared_ptr<USMAllocInfo>>
    findAllocInfoByAddress(uptr Address);

  private:
    ur_result_t updateShadowMemory(std::shared_ptr<ContextInfo> &ContextInfo,
                                   std::shared_ptr<DeviceInfo> &DeviceInfo,
                                   ur_queue_handle_t Queue);
    ur_result_t enqueueAllocInfo(ur_context_handle_t Context,
                                 std::shared_ptr<DeviceInfo> &DeviceInfo,
                                 ur_queue_handle_t Queue,
                                 std::shared_ptr<USMAllocInfo> &AllocInfo);

    /// Initialize Global Variables & Kernel Name at first Launch
    ur_result_t prepareLaunch(ur_context_handle_t Context,
                              std::shared_ptr<DeviceInfo> &DeviceInfo,
                              ur_queue_handle_t Queue,
                              ur_kernel_handle_t Kernel, LaunchInfo &LaunchInfo,
                              uint32_t numWorkgroup);

    ur_result_t allocShadowMemory(ur_context_handle_t Context,
                                  std::shared_ptr<DeviceInfo> &DeviceInfo);
    ur_result_t enqueueMemSetShadow(ur_context_handle_t Context,
                                    std::shared_ptr<DeviceInfo> &DeviceInfo,
                                    ur_queue_handle_t Queue, uptr Addr,
                                    uptr Size, u8 Value);

    std::shared_ptr<ContextInfo> getContextInfo(ur_context_handle_t Context) {
        std::shared_lock<ur_shared_mutex> Guard(m_ContextMapMutex);
        assert(m_ContextMap.find(Context) != m_ContextMap.end());
        return m_ContextMap[Context];
    }

    std::shared_ptr<DeviceInfo> getDeviceInfo(ur_device_handle_t Device) {
        std::shared_lock<ur_shared_mutex> Guard(m_DeviceMapMutex);
        assert(m_DeviceMap.find(Device) != m_DeviceMap.end());
        return m_DeviceMap[Device];
    }

  private:
    std::unordered_map<ur_context_handle_t, std::shared_ptr<ContextInfo>>
        m_ContextMap;
    ur_shared_mutex m_ContextMapMutex;

    std::unordered_map<ur_device_handle_t, std::shared_ptr<DeviceInfo>>
        m_DeviceMap;
    ur_shared_mutex m_DeviceMapMutex;

    struct USMAllocInfoCompare {
        bool operator()(const std::shared_ptr<USMAllocInfo> &lhs,
                        const std::shared_ptr<USMAllocInfo> &rhs) const {
            auto p1 = std::make_pair(lhs->AllocBegin,
                                     lhs->AllocBegin + lhs->AllocSize);
            auto p2 = std::make_pair(rhs->AllocBegin,
                                     rhs->AllocBegin + rhs->AllocSize);
            return p1 < p2;
        }
    };

    using AllocaionRangSet =
        std::multiset<std::shared_ptr<USMAllocInfo>, USMAllocInfoCompare>;

    AllocaionRangSet m_AllocationsMap;
    ur_shared_mutex m_AllocationsMapMutex;

    uint32_t cl_Debug = 0;
    uint32_t cl_MaxQuarantineSizeMB = 0;

    std::unique_ptr<Quarantine> m_Quarantine;
};

} // namespace ur_sanitizer_layer
