/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_interceptor.hpp
 *
 */

#pragma once

#include "msan_allocator.hpp"
#include "msan_buffer.hpp"
#include "msan_libdevice.hpp"
#include "msan_shadow.hpp"
#include "sanitizer_common/sanitizer_common.hpp"
#include "sanitizer_common/sanitizer_options.hpp"
#include "ur_sanitizer_layer.hpp"

#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ur_sanitizer_layer {
namespace msan {

struct AllocInfoList {
  std::vector<std::shared_ptr<MsanAllocInfo>> List;
  ur_shared_mutex Mutex;
};

struct DeviceInfo {
  ur_device_handle_t Handle;

  DeviceType Type = DeviceType::UNKNOWN;
  size_t Alignment = 0;
  std::shared_ptr<MsanShadowMemory> Shadow;

  // Device features
  bool IsSupportSharedSystemUSM = false;

  // Device handles are special and alive in the whole process lifetime,
  // so we needn't retain&release here.
  explicit DeviceInfo(ur_device_handle_t Device) : Handle(Device) {}

  ur_result_t allocShadowMemory(ur_context_handle_t Context);
};

struct QueueInfo {
  ur_queue_handle_t Handle;

  // lock this mutex if following fields are accessed
  ur_shared_mutex Mutex;
  ur_event_handle_t LastEvent;

  explicit QueueInfo(ur_queue_handle_t Queue)
      : Handle(Queue), LastEvent(nullptr) {
    [[maybe_unused]] auto Result =
        getContext()->urDdiTable.Queue.pfnRetain(Queue);
    assert(Result == UR_RESULT_SUCCESS);
  }

  ~QueueInfo() {
    [[maybe_unused]] auto Result =
        getContext()->urDdiTable.Queue.pfnRelease(Handle);
    assert(Result == UR_RESULT_SUCCESS);
  }
};

struct KernelInfo {
  ur_kernel_handle_t Handle;
  std::atomic<int32_t> RefCount = 1;

  // sanitized kernel
  bool IsInstrumented = false;
  // check local memory
  bool IsCheckLocals = true;
  // check private memory
  bool IsCheckPrivates = true;

  // lock this mutex if following fields are accessed
  ur_shared_mutex Mutex;
  std::unordered_map<uint32_t, std::shared_ptr<MemBuffer>> BufferArgs;

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
};

struct ProgramInfo {
  ur_program_handle_t Handle;
  std::atomic<int32_t> RefCount = 1;

  struct KernelMetada {
    bool CheckLocals;
    bool CheckPrivates;
  };

  // Program is built only once, so we don't need to lock it
  std::unordered_map<std::string, KernelMetada> KernelMetadataMap;

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

  bool isKernelInstrumented(ur_kernel_handle_t Kernel) const;
  const KernelMetada &getKernelMetadata(ur_kernel_handle_t Kernel) const;
};

struct ContextInfo {
  ur_context_handle_t Handle;
  size_t MaxAllocatedSize = 1024;
  std::atomic<int32_t> RefCount = 1;

  std::vector<ur_device_handle_t> DeviceList;

  explicit ContextInfo(ur_context_handle_t Context) : Handle(Context) {
    [[maybe_unused]] auto Result =
        getContext()->urDdiTable.Context.pfnRetain(Context);
    assert(Result == UR_RESULT_SUCCESS);
  }

  ~ContextInfo();
};

struct USMLaunchInfo {
  MsanLaunchInfo *Data = nullptr;

  ur_context_handle_t Context = nullptr;
  ur_device_handle_t Device = nullptr;
  const size_t *GlobalWorkSize = nullptr;
  const size_t *GlobalWorkOffset = nullptr;
  std::vector<size_t> LocalWorkSize;
  uint32_t WorkDim = 0;

  USMLaunchInfo(ur_context_handle_t Context, ur_device_handle_t Device,
                const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
                const size_t *GlobalWorkOffset, uint32_t WorkDim)
      : Context(Context), Device(Device), GlobalWorkSize(GlobalWorkSize),
        GlobalWorkOffset(GlobalWorkOffset), WorkDim(WorkDim) {
    if (LocalWorkSize) {
      this->LocalWorkSize =
          std::vector<size_t>(LocalWorkSize, LocalWorkSize + WorkDim);
    }
  }
  ~USMLaunchInfo();

  ur_result_t initialize();
};

struct DeviceGlobalInfo {
  uptr Size;
  uptr Addr;
};

struct SpirKernelInfo {
  uptr KernelName;
  uptr Size;
  uptr CheckLocals;
  uptr CheckPrivates;
};

class MsanInterceptor {
public:
  explicit MsanInterceptor();

  ~MsanInterceptor();

  ur_result_t allocateMemory(ur_context_handle_t Context,
                             ur_device_handle_t Device,
                             const ur_usm_desc_t *Properties,
                             ur_usm_pool_handle_t Pool, size_t Size,
                             AllocType Type, void **ResultPtr);
  ur_result_t releaseMemory(ur_context_handle_t Context, void *Ptr);

  ur_result_t registerProgram(ur_program_handle_t Program);
  ur_result_t unregisterProgram(ur_program_handle_t Program);

  ur_result_t preLaunchKernel(ur_kernel_handle_t Kernel,
                              ur_queue_handle_t Queue,
                              msan::USMLaunchInfo &LaunchInfo);
  ur_result_t postLaunchKernel(ur_kernel_handle_t Kernel,
                               ur_queue_handle_t Queue,
                               msan::USMLaunchInfo &LaunchInfo);

  ur_result_t insertContext(ur_context_handle_t Context,
                            std::shared_ptr<msan::ContextInfo> &CI);
  ur_result_t eraseContext(ur_context_handle_t Context);

  ur_result_t insertDevice(ur_device_handle_t Device,
                           std::shared_ptr<msan::DeviceInfo> &CI);
  ur_result_t eraseDevice(ur_device_handle_t Device);

  ur_result_t insertProgram(ur_program_handle_t Program);
  ur_result_t eraseProgram(ur_program_handle_t Program);

  ur_result_t insertMemBuffer(std::shared_ptr<MemBuffer> MemBuffer);
  ur_result_t eraseMemBuffer(ur_mem_handle_t MemHandle);
  std::shared_ptr<MemBuffer> getMemBuffer(ur_mem_handle_t MemHandle);

  ur_result_t holdAdapter(ur_adapter_handle_t Adapter) {
    std::scoped_lock<ur_shared_mutex> Guard(m_AdaptersMutex);
    if (m_Adapters.find(Adapter) != m_Adapters.end()) {
      return UR_RESULT_SUCCESS;
    }
    UR_CALL(getContext()->urDdiTable.Global.pfnAdapterRetain(Adapter));
    m_Adapters.insert(Adapter);
    return UR_RESULT_SUCCESS;
  }

  std::optional<MsanAllocationIterator> findAllocInfoByAddress(uptr Address);

  std::vector<MsanAllocationIterator>
  findAllocInfoByContext(ur_context_handle_t Context);

  std::shared_ptr<msan::ContextInfo>
  getContextInfo(ur_context_handle_t Context) {
    std::shared_lock<ur_shared_mutex> Guard(m_ContextMapMutex);
    assert(m_ContextMap.find(Context) != m_ContextMap.end());
    return m_ContextMap[Context];
  }

  std::shared_ptr<msan::DeviceInfo> getDeviceInfo(ur_device_handle_t Device) {
    std::shared_lock<ur_shared_mutex> Guard(m_DeviceMapMutex);
    assert(m_DeviceMap.find(Device) != m_DeviceMap.end());
    return m_DeviceMap[Device];
  }

  std::shared_ptr<msan::ProgramInfo>
  getProgramInfo(ur_program_handle_t Program) {
    std::shared_lock<ur_shared_mutex> Guard(m_ProgramMapMutex);
    assert(m_ProgramMap.find(Program) != m_ProgramMap.end());
    return m_ProgramMap[Program];
  }

  KernelInfo &getOrCreateKernelInfo(ur_kernel_handle_t Kernel);
  ur_result_t eraseKernelInfo(ur_kernel_handle_t Kernel);

  void exitWithErrors() {
    m_NormalExit = false;
    exit(1);
  }

  bool isNormalExit() { return m_NormalExit; }

private:
  /// Initialize Global Variables & Kernel Name at first Launch
  ur_result_t prepareLaunch(std::shared_ptr<msan::DeviceInfo> &DeviceInfo,
                            ur_queue_handle_t Queue, ur_kernel_handle_t Kernel,
                            msan::USMLaunchInfo &LaunchInfo);

  ur_result_t allocShadowMemory(ur_context_handle_t Context,
                                std::shared_ptr<msan::DeviceInfo> &DeviceInfo);

  ur_result_t registerSpirKernels(ur_program_handle_t Program);
  ur_result_t registerDeviceGlobals(ur_program_handle_t Program);

private:
  std::unordered_map<ur_context_handle_t, std::shared_ptr<msan::ContextInfo>>
      m_ContextMap;
  ur_shared_mutex m_ContextMapMutex;
  std::unordered_map<ur_device_handle_t, std::shared_ptr<msan::DeviceInfo>>
      m_DeviceMap;
  ur_shared_mutex m_DeviceMapMutex;

  std::unordered_map<ur_program_handle_t, std::shared_ptr<msan::ProgramInfo>>
      m_ProgramMap;
  ur_shared_mutex m_ProgramMapMutex;

  std::unordered_map<ur_kernel_handle_t, std::shared_ptr<msan::KernelInfo>>
      m_KernelMap;
  ur_shared_mutex m_KernelMapMutex;

  std::unordered_map<ur_mem_handle_t, std::shared_ptr<MemBuffer>>
      m_MemBufferMap;
  ur_shared_mutex m_MemBufferMapMutex;

  /// Assumption: all USM chunks are allocated in one VA
  MsanAllocationMap m_AllocationMap;
  ur_shared_mutex m_AllocationMapMutex;

  std::unordered_set<ur_adapter_handle_t> m_Adapters;
  ur_shared_mutex m_AdaptersMutex;

  bool m_NormalExit = true;
};

} // namespace msan

msan::MsanInterceptor *getMsanInterceptor();

} // namespace ur_sanitizer_layer
