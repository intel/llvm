/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file tsan_interceptor.hpp
 *
 */

#pragma once

#include "sanitizer_common/sanitizer_allocator.hpp"
#include "sanitizer_common/sanitizer_common.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "tsan_buffer.hpp"
#include "tsan_libdevice.hpp"
#include "tsan_shadow.hpp"
#include "ur_sanitizer_layer.hpp"

#include <unordered_map>
#include <unordered_set>

namespace ur_sanitizer_layer {
namespace tsan {

struct TsanAllocInfo {
  uptr AllocBegin = 0;

  size_t AllocSize = 0;

  bool operator<(TsanAllocInfo const &Other) const {
    return AllocBegin < Other.AllocBegin;
  }
};

struct DeviceInfo {
  ur_device_handle_t Handle;

  DeviceType Type = DeviceType::UNKNOWN;

  std::shared_ptr<ShadowMemory> Shadow;

  ur_shared_mutex AllocInfosMutex;
  std::set<TsanAllocInfo> AllocInfos;

  explicit DeviceInfo(ur_device_handle_t Device) : Handle(Device) {}

  ur_result_t allocShadowMemory();

  void insertAllocInfo(TsanAllocInfo AI);
};

struct ContextInfo {
  ur_context_handle_t Handle;

  std::atomic<uint32_t> RefCount = 1;

  std::vector<ur_device_handle_t> DeviceList;

  ur_shared_mutex InternalQueueMapMutex;
  std::unordered_map<ur_device_handle_t, std::optional<ManagedQueue>>
      InternalQueueMap;

  explicit ContextInfo(ur_context_handle_t Context) : Handle(Context) {
    [[maybe_unused]] auto Result =
        getContext()->urDdiTable.Context.pfnRetain(Context);
    assert(Result == UR_RESULT_SUCCESS);
  }

  ~ContextInfo() {
    InternalQueueMap.clear();
    [[maybe_unused]] auto Result =
        getContext()->urDdiTable.Context.pfnRelease(Handle);
    assert(Result == UR_RESULT_SUCCESS);
  }

  ContextInfo(const ContextInfo &) = delete;

  ContextInfo &operator=(const ContextInfo &) = delete;

  ur_queue_handle_t getInternalQueue(ur_device_handle_t);
};

struct DeviceGlobalInfo {
  uptr Size;
  uptr Addr;
};

struct ProgramInfo {
  ur_program_handle_t Handle;
  std::atomic<int32_t> RefCount = 1;

  // Program is built only once, so we don't need to lock it
  std::vector<TsanAllocInfo> AllocInfoForGlobals;

  ProgramInfo() = default;

  explicit ProgramInfo(ur_program_handle_t Program) : Handle(Program) {
    [[maybe_unused]] auto Result =
        getContext()->urDdiTable.Program.pfnRetain(Handle);
    assert(Result == UR_RESULT_SUCCESS);
  }

  ~ProgramInfo() {
    [[maybe_unused]] auto Result =
        getContext()->urDdiTable.Program.pfnRelease(Handle);
    assert(Result == UR_RESULT_SUCCESS);
  }

  ProgramInfo(const ProgramInfo &) = delete;

  ProgramInfo &operator=(const ProgramInfo &) = delete;
};

struct KernelInfo {
  ur_kernel_handle_t Handle = nullptr;
  std::atomic<int32_t> RefCount = 1;

  // lock this mutex if following fields are accessed
  ur_shared_mutex Mutex;
  std::unordered_map<uint32_t, std::shared_ptr<MemBuffer>> BufferArgs;

  // Need preserve the order of local arguments
  std::map<uint32_t, TsanLocalArgsInfo> LocalArgs;

  KernelInfo() = default;

  explicit KernelInfo(ur_kernel_handle_t Kernel) : Handle(Kernel) {
    [[maybe_unused]] auto Result =
        getContext()->urDdiTable.Kernel.pfnRetain(Kernel);
    assert(Result == UR_RESULT_SUCCESS);
  }

  ~KernelInfo() {
    [[maybe_unused]] auto Result =
        getContext()->urDdiTable.Kernel.pfnRelease(Handle);
    assert(Result == UR_RESULT_SUCCESS);
  }

  KernelInfo(const KernelInfo &) = delete;

  KernelInfo &operator=(const KernelInfo &) = delete;
};

struct TsanRuntimeDataWrapper {
  TsanRuntimeData Host{};

  TsanRuntimeData *DevicePtr = nullptr;

  ur_context_handle_t Context{};

  ur_device_handle_t Device{};

  TsanRuntimeDataWrapper(ur_context_handle_t Context, ur_device_handle_t Device)
      : Context(Context), Device(Device) {}

  ~TsanRuntimeDataWrapper();

  TsanRuntimeDataWrapper(const TsanRuntimeDataWrapper &) = delete;

  TsanRuntimeDataWrapper &operator=(const TsanRuntimeDataWrapper &) = delete;

  TsanRuntimeData *getDevicePtr();

  ur_result_t syncFromDevice(ur_queue_handle_t Queue);

  ur_result_t syncToDevice(ur_queue_handle_t Queue);

  bool hasReport(ur_queue_handle_t Queue);

  ur_result_t
  importLocalArgsInfo(ur_queue_handle_t Queue,
                      const std::vector<TsanLocalArgsInfo> &LocalArgs);
};

struct LaunchInfo {
  ur_context_handle_t Context = nullptr;
  ur_device_handle_t Device = nullptr;
  const size_t *GlobalWorkSize = nullptr;
  std::vector<size_t> GlobalWorkOffset;
  std::vector<size_t> LocalWorkSize;
  uint32_t WorkDim = 0;
  TsanRuntimeDataWrapper Data;

  LaunchInfo(ur_context_handle_t Context, ur_device_handle_t Device,
             const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
             const size_t *GlobalWorkOffset, uint32_t WorkDim)
      : Context(Context), Device(Device), GlobalWorkSize(GlobalWorkSize),
        WorkDim(WorkDim), Data(Context, Device) {
    [[maybe_unused]] auto Result =
        getContext()->urDdiTable.Context.pfnRetain(Context);
    assert(Result == UR_RESULT_SUCCESS);
    Result = getContext()->urDdiTable.Device.pfnRetain(Device);
    assert(Result == UR_RESULT_SUCCESS);
    if (LocalWorkSize) {
      this->LocalWorkSize =
          std::vector<size_t>(LocalWorkSize, LocalWorkSize + WorkDim);
    }
    // UR doesn't allow GlobalWorkOffset is null, we need to construct a zero
    // value array if user doesn't specify its value.
    if (GlobalWorkOffset) {
      this->GlobalWorkOffset =
          std::vector<size_t>(GlobalWorkOffset, GlobalWorkOffset + WorkDim);
    } else {
      this->GlobalWorkOffset = std::vector<size_t>(WorkDim, 0);
    }
  }

  ~LaunchInfo() {
    [[maybe_unused]] ur_result_t Result;
    Result = getContext()->urDdiTable.Context.pfnRelease(Context);
    assert(Result == UR_RESULT_SUCCESS);
    Result = getContext()->urDdiTable.Device.pfnRelease(Device);
    assert(Result == UR_RESULT_SUCCESS);
  }

  LaunchInfo(const LaunchInfo &) = delete;

  LaunchInfo &operator=(const LaunchInfo &) = delete;
};

class TsanInterceptor {
public:
  ~TsanInterceptor();

  ur_result_t allocateMemory(ur_context_handle_t Context,
                             ur_device_handle_t Device,
                             const ur_usm_desc_t *Properties,
                             ur_usm_pool_handle_t Pool, size_t Size,
                             AllocType Type, void **ResultPtr);

  ur_result_t releaseMemory(ur_context_handle_t Context, void *Ptr);

  ur_result_t registerProgram(ur_program_handle_t Program);

  ur_result_t unregisterProgram(ur_program_handle_t Program);

  ur_result_t insertContext(ur_context_handle_t Context,
                            std::shared_ptr<ContextInfo> &CI);

  ur_result_t eraseContext(ur_context_handle_t Context);

  ur_result_t insertDevice(ur_device_handle_t Device,
                           std::shared_ptr<DeviceInfo> &DI);

  ur_result_t insertProgram(ur_program_handle_t Program);

  ur_result_t eraseProgram(ur_program_handle_t Program);

  ur_result_t insertKernel(ur_kernel_handle_t Kernel);

  ur_result_t eraseKernel(ur_kernel_handle_t Kernel);

  ur_result_t insertMemBuffer(std::shared_ptr<MemBuffer> MemBuffer);

  ur_result_t eraseMemBuffer(ur_mem_handle_t MemHandle);

  std::shared_ptr<MemBuffer> getMemBuffer(ur_mem_handle_t MemHandle);

  ur_result_t holdAdapter(ur_adapter_handle_t Adapter) {
    std::scoped_lock<ur_shared_mutex> Guard(m_AdaptersMutex);
    if (m_Adapters.find(Adapter) != m_Adapters.end()) {
      return UR_RESULT_SUCCESS;
    }
    UR_CALL(getContext()->urDdiTable.Adapter.pfnRetain(Adapter));
    m_Adapters.insert(Adapter);
    return UR_RESULT_SUCCESS;
  }

  ur_result_t preLaunchKernel(ur_kernel_handle_t Kernel,
                              ur_queue_handle_t Queue, LaunchInfo &LaunchInfo);

  ur_result_t postLaunchKernel(ur_kernel_handle_t Kernel,
                               ur_queue_handle_t Queue, LaunchInfo &LaunchInfo);

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

  ProgramInfo &getProgramInfo(ur_program_handle_t Program) {
    std::shared_lock<ur_shared_mutex> Guard(m_ProgramMapMutex);
    assert(m_ProgramMap.find(Program) != m_ProgramMap.end());
    return m_ProgramMap[Program];
  }

  KernelInfo &getKernelInfo(ur_kernel_handle_t Kernel) {
    std::shared_lock<ur_shared_mutex> Guard(m_KernelMapMutex);
    assert(m_KernelMap.find(Kernel) != m_KernelMap.end());
    return m_KernelMap[Kernel];
  }

  ur_shared_mutex KernelLaunchMutex;

private:
  ur_result_t updateShadowMemory(std::shared_ptr<DeviceInfo> &DI,
                                 ur_kernel_handle_t Kernel,
                                 ur_queue_handle_t Queue);

  ur_result_t prepareLaunch(std::shared_ptr<ContextInfo> &CI,
                            std::shared_ptr<DeviceInfo> &DI,
                            ur_queue_handle_t Queue, ur_kernel_handle_t Kernel,
                            LaunchInfo &LaunchInfo);

  ur_result_t registerDeviceGlobals(ur_program_handle_t Program);

private:
  std::unordered_map<ur_context_handle_t, std::shared_ptr<ContextInfo>>
      m_ContextMap;
  ur_shared_mutex m_ContextMapMutex;

  std::unordered_map<ur_device_handle_t, std::shared_ptr<DeviceInfo>>
      m_DeviceMap;
  ur_shared_mutex m_DeviceMapMutex;

  std::unordered_map<ur_program_handle_t, ProgramInfo> m_ProgramMap;
  ur_shared_mutex m_ProgramMapMutex;

  std::unordered_map<ur_kernel_handle_t, KernelInfo> m_KernelMap;
  ur_shared_mutex m_KernelMapMutex;

  std::unordered_map<ur_mem_handle_t, std::shared_ptr<MemBuffer>>
      m_MemBufferMap;
  ur_shared_mutex m_MemBufferMapMutex;

  std::unordered_set<ur_adapter_handle_t> m_Adapters;
  ur_shared_mutex m_AdaptersMutex;
};

} // namespace tsan

tsan::TsanInterceptor *getTsanInterceptor();

} // namespace ur_sanitizer_layer
