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
#include "asan_libdevice.hpp"
#include "common.hpp"
#include "ur_sanitizer_layer.hpp"

#include <memory>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace ur_sanitizer_layer {

class Quarantine;

struct AllocInfoList {
    std::vector<std::shared_ptr<AllocInfo>> List;
    ur_shared_mutex Mutex;
};

struct DeviceInfo {
    ur_device_handle_t Handle;

    DeviceType Type = DeviceType::UNKNOWN;
    size_t Alignment = 0;
    uptr ShadowOffset = 0;
    uptr ShadowOffsetEnd = 0;

    ur_mutex Mutex;
    std::queue<std::shared_ptr<AllocInfo>> Quarantine;
    size_t QuarantineSize = 0;

    explicit DeviceInfo(ur_device_handle_t Device) : Handle(Device) {
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

    ur_shared_mutex Mutex;
    ur_event_handle_t LastEvent;

    explicit QueueInfo(ur_queue_handle_t Queue)
        : Handle(Queue), LastEvent(nullptr) {
        [[maybe_unused]] auto Result =
            context.urDdiTable.Queue.pfnRetain(Queue);
        assert(Result == UR_RESULT_SUCCESS);
    }

    ~QueueInfo() {
        [[maybe_unused]] auto Result =
            context.urDdiTable.Queue.pfnRelease(Handle);
        assert(Result == UR_RESULT_SUCCESS);
    }
};

struct ContextInfo {
    ur_context_handle_t Handle;

    std::vector<ur_device_handle_t> DeviceList;
    std::unordered_map<ur_device_handle_t, AllocInfoList> AllocInfosMap;

    explicit ContextInfo(ur_context_handle_t Context) : Handle(Context) {
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
                         std::shared_ptr<AllocInfo> &AI) {
        for (auto Device : Devices) {
            auto &AllocInfos = AllocInfosMap[Device];
            std::scoped_lock<ur_shared_mutex> Guard(AllocInfos.Mutex);
            AllocInfos.List.emplace_back(AI);
        }
    }
};

struct LaunchInfo {
    uptr LocalShadowOffset = 0;
    uptr LocalShadowOffsetEnd = 0;
    DeviceSanitizerReport SPIR_DeviceSanitizerReportMem;

    ur_context_handle_t Context = nullptr;
    const size_t *GlobalWorkSize = nullptr;
    const size_t *GlobalWorkOffset = nullptr;
    std::vector<size_t> LocalWorkSize;
    uint32_t WorkDim = 0;

    LaunchInfo(ur_context_handle_t Context, const size_t *GlobalWorkSize,
               const size_t *LocalWorkSize, const size_t *GlobalWorkOffset,
               uint32_t WorkDim)
        : Context(Context), GlobalWorkSize(GlobalWorkSize),
          GlobalWorkOffset(GlobalWorkOffset), WorkDim(WorkDim) {
        [[maybe_unused]] auto Result =
            context.urDdiTable.Context.pfnRetain(Context);
        assert(Result == UR_RESULT_SUCCESS);
        if (LocalWorkSize) {
            this->LocalWorkSize =
                std::vector<size_t>(LocalWorkSize, LocalWorkSize + WorkDim);
        }
    }
    ~LaunchInfo();
};

struct DeviceGlobalInfo {
    uptr Size;
    uptr SizeWithRedZone;
    uptr Addr;
};

class SanitizerInterceptor {
  public:
    explicit SanitizerInterceptor();

    ~SanitizerInterceptor();

    ur_result_t allocateMemory(ur_context_handle_t Context,
                               ur_device_handle_t Device,
                               const ur_usm_desc_t *Properties,
                               ur_usm_pool_handle_t Pool, size_t Size,
                               AllocType Type, void **ResultPtr);
    ur_result_t releaseMemory(ur_context_handle_t Context, void *Ptr);

    ur_result_t registerDeviceGlobals(ur_context_handle_t Context,
                                      ur_program_handle_t Program);

    ur_result_t preLaunchKernel(ur_kernel_handle_t Kernel,
                                ur_queue_handle_t Queue,
                                LaunchInfo &LaunchInfo);

    ur_result_t postLaunchKernel(ur_kernel_handle_t Kernel,
                                 ur_queue_handle_t Queue,
                                 ur_event_handle_t &Event,
                                 LaunchInfo &LaunchInfo);

    ur_result_t insertContext(ur_context_handle_t Context,
                              std::shared_ptr<ContextInfo> &CI);
    ur_result_t eraseContext(ur_context_handle_t Context);

    ur_result_t insertDevice(ur_device_handle_t Device,
                             std::shared_ptr<DeviceInfo> &CI);
    ur_result_t eraseDevice(ur_device_handle_t Device);

    std::optional<AllocationIterator> findAllocInfoByAddress(uptr Address);

    std::shared_ptr<ContextInfo> getContextInfo(ur_context_handle_t Context) {
        std::shared_lock<ur_shared_mutex> Guard(m_ContextMapMutex);
        assert(m_ContextMap.find(Context) != m_ContextMap.end());
        return m_ContextMap[Context];
    }

  private:
    ur_result_t updateShadowMemory(std::shared_ptr<ContextInfo> &ContextInfo,
                                   std::shared_ptr<DeviceInfo> &DeviceInfo,
                                   ur_queue_handle_t Queue);
    ur_result_t enqueueAllocInfo(ur_context_handle_t Context,
                                 std::shared_ptr<DeviceInfo> &DeviceInfo,
                                 ur_queue_handle_t Queue,
                                 std::shared_ptr<AllocInfo> &AI);

    /// Initialize Global Variables & Kernel Name at first Launch
    ur_result_t prepareLaunch(ur_context_handle_t Context,
                              std::shared_ptr<DeviceInfo> &DeviceInfo,
                              ur_queue_handle_t Queue,
                              ur_kernel_handle_t Kernel,
                              LaunchInfo &LaunchInfo);

    ur_result_t allocShadowMemory(ur_context_handle_t Context,
                                  std::shared_ptr<DeviceInfo> &DeviceInfo);

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

    /// Assumption: all USM chunks are allocated in one VA
    AllocationMap m_AllocationMap;
    ur_shared_mutex m_AllocationMapMutex;

    // We use "uint64_t" here because EnqueueWriteGlobal will fail when it's "uint32_t"
    uint64_t cl_Debug = 0;
    uint32_t cl_MaxQuarantineSizeMB = 0;
    bool cl_DetectLocals = true;

    std::unique_ptr<Quarantine> m_Quarantine;
};

} // namespace ur_sanitizer_layer
