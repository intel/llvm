//===----------------------------------------------------------------------===//
/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file tsan_interceptor.cpp
 *
 */

#include "tsan_interceptor.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "tsan_report.hpp"

namespace ur_sanitizer_layer {
namespace tsan {

TsanRuntimeDataWrapper::~TsanRuntimeDataWrapper() {
  [[maybe_unused]] ur_result_t Result;
  if (Host.LocalArgs) {
    Result =
        getContext()->urDdiTable.USM.pfnFree(Context, (void *)Host.LocalArgs);
    assert(Result == UR_RESULT_SUCCESS);
  }
  if (DevicePtr) {
    Result = getContext()->urDdiTable.USM.pfnFree(Context, DevicePtr);
    assert(Result == UR_RESULT_SUCCESS);
  }
}

TsanRuntimeData *TsanRuntimeDataWrapper::getDevicePtr() {
  if (DevicePtr == nullptr) {
    ur_result_t Result = getContext()->urDdiTable.USM.pfnDeviceAlloc(
        Context, Device, nullptr, nullptr, sizeof(TsanRuntimeData),
        (void **)&DevicePtr);
    if (Result != UR_RESULT_SUCCESS) {
      UR_LOG(ERR, "Failed to alloc device usm for asan runtime data: {}",
             Result);
    }
  }
  return DevicePtr;
}

ur_result_t TsanRuntimeDataWrapper::syncFromDevice(ur_queue_handle_t Queue) {
  UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
      Queue, true, ur_cast<void *>(&Host), getDevicePtr(),
      sizeof(TsanRuntimeData), 0, nullptr, nullptr));

  return UR_RESULT_SUCCESS;
}

ur_result_t TsanRuntimeDataWrapper::syncToDevice(ur_queue_handle_t Queue) {
  UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
      Queue, true, getDevicePtr(), ur_cast<void *>(&Host),
      sizeof(TsanRuntimeData), 0, nullptr, nullptr));

  return UR_RESULT_SUCCESS;
}

ur_result_t TsanRuntimeDataWrapper::importLocalArgsInfo(
    ur_queue_handle_t Queue, const std::vector<TsanLocalArgsInfo> &LocalArgs) {
  assert(!LocalArgs.empty());

  Host.NumLocalArgs = LocalArgs.size();
  const size_t LocalArgsInfoSize =
      sizeof(TsanLocalArgsInfo) * Host.NumLocalArgs;
  UR_CALL(getContext()->urDdiTable.USM.pfnDeviceAlloc(
      Context, Device, nullptr, nullptr, LocalArgsInfoSize,
      ur_cast<void **>(&Host.LocalArgs)));

  UR_CALL(getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
      Queue, true, Host.LocalArgs, &LocalArgs[0], LocalArgsInfoSize, 0, nullptr,
      nullptr));

  return UR_RESULT_SUCCESS;
}

ur_result_t DeviceInfo::allocShadowMemory() {
  ur_context_handle_t ShadowContext;
  UR_CALL(getContext()->urDdiTable.Context.pfnCreate(1, &Handle, nullptr,
                                                     &ShadowContext));
  Shadow = GetShadowMemory(ShadowContext, Handle, Type);
  assert(Shadow && "Failed to get shadow memory");
  UR_CALL(Shadow->Setup());
  UR_LOG_L(getContext()->logger, INFO, "ShadowMemory(Global): {} - {}",
           (void *)Shadow->ShadowBegin, (void *)Shadow->ShadowEnd);
  return UR_RESULT_SUCCESS;
}

void ContextInfo::insertAllocInfo(ur_device_handle_t Device, TsanAllocInfo AI) {
  if (Device) {
    std::scoped_lock<ur_shared_mutex> Guard(AllocInfosMapMutex);
    AllocInfosMap[Device].emplace_back(std::move(AI));
  } else {
    for (auto Device : DeviceList) {
      std::scoped_lock<ur_shared_mutex> Guard(AllocInfosMapMutex);
      AllocInfosMap[Device].emplace_back(AI);
    }
  }
}

ur_result_t TsanInterceptor::allocateMemory(ur_context_handle_t Context,
                                            ur_device_handle_t Device,
                                            const ur_usm_desc_t *Properties,
                                            ur_usm_pool_handle_t Pool,
                                            size_t Size, AllocType Type,
                                            void **ResultPtr) {
  auto CI = getContextInfo(Context);

  void *Allocated = nullptr;

  if (Type == AllocType::DEVICE_USM) {
    UR_CALL(getContext()->urDdiTable.USM.pfnDeviceAlloc(
        Context, Device, Properties, Pool, Size, &Allocated));
  } else if (Type == AllocType::HOST_USM) {
    UR_CALL(getContext()->urDdiTable.USM.pfnHostAlloc(Context, Properties, Pool,
                                                      Size, &Allocated));
  } else if (Type == AllocType::SHARED_USM) {
    UR_CALL(getContext()->urDdiTable.USM.pfnSharedAlloc(
        Context, Device, Properties, Pool, Size, &Allocated));
  }

  auto AI = TsanAllocInfo{reinterpret_cast<uptr>(Allocated), Size};
  // For updating shadow memory
  CI->insertAllocInfo(Device, std::move(AI));

  *ResultPtr = Allocated;
  return UR_RESULT_SUCCESS;
}

ur_result_t TsanInterceptor::registerProgram(ur_program_handle_t Program) {
  UR_LOG_L(getContext()->logger, INFO, "registerDeviceGlobals");
  UR_CALL(registerDeviceGlobals(Program));
  return UR_RESULT_SUCCESS;
}

ur_result_t
TsanInterceptor::registerDeviceGlobals(ur_program_handle_t Program) {
  std::vector<ur_device_handle_t> Devices = GetDevices(Program);
  assert(Devices.size() != 0 && "No devices in registerDeviceGlobals");
  auto Context = GetContext(Program);
  auto ContextInfo = getContextInfo(Context);

  for (auto Device : Devices) {
    ManagedQueue Queue(Context, Device);

    size_t MetadataSize;
    void *MetadataPtr;
    auto Result = getContext()->urDdiTable.Program.pfnGetGlobalVariablePointer(
        Device, Program, kSPIR_TsanDeviceGlobalMetadata, &MetadataSize,
        &MetadataPtr);
    if (Result != UR_RESULT_SUCCESS) {
      UR_LOG_L(getContext()->logger, INFO, "No device globals");
      continue;
    }

    const uint64_t NumOfDeviceGlobal = MetadataSize / sizeof(DeviceGlobalInfo);
    assert((MetadataSize % sizeof(DeviceGlobalInfo) == 0) &&
           "DeviceGlobal metadata size is not correct");
    std::vector<DeviceGlobalInfo> GVInfos(NumOfDeviceGlobal);
    Result = getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
        Queue, true, &GVInfos[0], MetadataPtr,
        sizeof(DeviceGlobalInfo) * NumOfDeviceGlobal, 0, nullptr, nullptr);
    if (Result != UR_RESULT_SUCCESS) {
      UR_LOG_L(getContext()->logger, ERR, "Device Global[{}] Read Failed: {}",
               kSPIR_TsanDeviceGlobalMetadata, Result);
      return Result;
    }

    for (size_t i = 0; i < NumOfDeviceGlobal; i++) {
      const auto &GVInfo = GVInfos[i];
      auto AI = TsanAllocInfo{GVInfo.Addr, GVInfo.Size};
      ContextInfo->insertAllocInfo(Device, std::move(AI));
    }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t TsanInterceptor::insertContext(ur_context_handle_t Context,
                                           std::shared_ptr<ContextInfo> &CI) {
  std::scoped_lock<ur_shared_mutex> Guard(m_ContextMapMutex);

  if (m_ContextMap.find(Context) != m_ContextMap.end()) {
    CI = m_ContextMap.at(Context);
    return UR_RESULT_SUCCESS;
  }

  CI = std::make_shared<ContextInfo>(Context);

  // Don't move CI, since it's a return value as well
  m_ContextMap.emplace(Context, CI);

  return UR_RESULT_SUCCESS;
}

ur_result_t TsanInterceptor::eraseContext(ur_context_handle_t Context) {
  std::scoped_lock<ur_shared_mutex> Guard(m_ContextMapMutex);
  assert(m_ContextMap.find(Context) != m_ContextMap.end());
  m_ContextMap.erase(Context);
  // TODO: Remove devices in each context
  return UR_RESULT_SUCCESS;
}

ur_result_t TsanInterceptor::insertKernel(ur_kernel_handle_t Kernel) {
  std::scoped_lock<ur_shared_mutex> Guard(m_KernelMapMutex);
  if (m_KernelMap.find(Kernel) != m_KernelMap.end()) {
    return UR_RESULT_SUCCESS;
  }

  m_KernelMap.emplace(Kernel, Kernel);

  return UR_RESULT_SUCCESS;
}

ur_result_t TsanInterceptor::eraseKernel(ur_kernel_handle_t Kernel) {
  std::scoped_lock<ur_shared_mutex> Guard(m_KernelMapMutex);
  assert(m_KernelMap.find(Kernel) != m_KernelMap.end());
  m_KernelMap.erase(Kernel);
  return UR_RESULT_SUCCESS;
}

ur_result_t TsanInterceptor::insertDevice(ur_device_handle_t Device,
                                          std::shared_ptr<DeviceInfo> &DI) {
  std::scoped_lock<ur_shared_mutex> Guard(m_DeviceMapMutex);

  if (m_DeviceMap.find(Device) != m_DeviceMap.end()) {
    DI = m_DeviceMap.at(Device);
    return UR_RESULT_SUCCESS;
  }

  DI = std::make_shared<DeviceInfo>(Device);

  // Don't move DI, since it's a return value as well
  m_DeviceMap.emplace(Device, DI);

  return UR_RESULT_SUCCESS;
}

ur_result_t
TsanInterceptor::insertMemBuffer(std::shared_ptr<MemBuffer> MemBuffer) {
  std::scoped_lock<ur_shared_mutex> Guard(m_MemBufferMapMutex);
  assert(m_MemBufferMap.find(ur_cast<ur_mem_handle_t>(MemBuffer.get())) ==
         m_MemBufferMap.end());
  m_MemBufferMap.emplace(reinterpret_cast<ur_mem_handle_t>(MemBuffer.get()),
                         MemBuffer);
  return UR_RESULT_SUCCESS;
}

ur_result_t TsanInterceptor::eraseMemBuffer(ur_mem_handle_t MemHandle) {
  std::scoped_lock<ur_shared_mutex> Guard(m_MemBufferMapMutex);
  assert(m_MemBufferMap.find(MemHandle) != m_MemBufferMap.end());
  m_MemBufferMap.erase(MemHandle);
  return UR_RESULT_SUCCESS;
}

std::shared_ptr<MemBuffer>
TsanInterceptor::getMemBuffer(ur_mem_handle_t MemHandle) {
  std::shared_lock<ur_shared_mutex> Guard(m_MemBufferMapMutex);
  if (m_MemBufferMap.find(MemHandle) != m_MemBufferMap.end()) {
    return m_MemBufferMap[MemHandle];
  }
  return nullptr;
}

ur_result_t TsanInterceptor::preLaunchKernel(ur_kernel_handle_t Kernel,
                                             ur_queue_handle_t Queue,
                                             LaunchInfo &LaunchInfo) {
  auto CI = getContextInfo(GetContext(Queue));
  auto DI = getDeviceInfo(GetDevice(Queue));

  ManagedQueue InternalQueue(CI->Handle, DI->Handle);
  if (!InternalQueue) {
    UR_LOG_L(getContext()->logger, ERR, "Failed to create internal queue");
    return UR_RESULT_ERROR_INVALID_QUEUE;
  }

  UR_CALL(prepareLaunch(CI, DI, InternalQueue, Kernel, LaunchInfo));

  UR_CALL(updateShadowMemory(CI, DI, InternalQueue));

  return UR_RESULT_SUCCESS;
}

ur_result_t TsanInterceptor::postLaunchKernel(ur_kernel_handle_t Kernel,
                                              ur_queue_handle_t Queue,
                                              LaunchInfo &LaunchInfo) {
  // FIXME: We must use block operation here, until we support
  // urEventSetCallback
  UR_CALL(getContext()->urDdiTable.Queue.pfnFinish(Queue));

  UR_CALL(LaunchInfo.Data.syncFromDevice(Queue));

  for (uptr ReportIndex = 0;
       ReportIndex < LaunchInfo.Data.Host.RecordedReportCount; ReportIndex++) {
    ReportDataRace(LaunchInfo.Data.Host.Report[ReportIndex], Kernel);
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t TsanInterceptor::prepareLaunch(std::shared_ptr<ContextInfo> &,
                                           std::shared_ptr<DeviceInfo> &DI,
                                           ur_queue_handle_t Queue,
                                           ur_kernel_handle_t Kernel,
                                           LaunchInfo &LaunchInfo) {
  // Set membuffer arguments
  auto &KernelInfo = getKernelInfo(Kernel);
  {
    std::shared_lock<ur_shared_mutex> Guard(KernelInfo.Mutex);
    for (const auto &[ArgIndex, MemBuffer] : KernelInfo.BufferArgs) {
      char *ArgPointer = nullptr;
      UR_CALL(MemBuffer->getHandle(DI->Handle, ArgPointer));
      ur_result_t URes = getContext()->urDdiTable.Kernel.pfnSetArgPointer(
          Kernel, ArgIndex, nullptr, ArgPointer);
      if (URes != UR_RESULT_SUCCESS) {
        UR_LOG_L(getContext()->logger, ERR,
                 "Failed to set buffer {} as the {} arg to kernel {}: {}",
                 ur_cast<ur_mem_handle_t>(MemBuffer.get()), ArgIndex, Kernel,
                 URes);
      }
    }
  }

  // Get suggested local work size is user doesn't determine it.
  if (LaunchInfo.LocalWorkSize.empty()) {
    LaunchInfo.LocalWorkSize.resize(LaunchInfo.WorkDim);
    auto URes = getContext()->urDdiTable.Kernel.pfnGetSuggestedLocalWorkSize(
        Kernel, Queue, LaunchInfo.WorkDim, LaunchInfo.GlobalWorkOffset,
        LaunchInfo.GlobalWorkSize, LaunchInfo.LocalWorkSize.data());
    if (URes != UR_RESULT_SUCCESS) {
      if (URes != UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
        return URes;
      }
      // If urKernelGetSuggestedLocalWorkSize is not supported by driver, we
      // fallback to inefficient implementation
      for (size_t Dim = 0; Dim < LaunchInfo.WorkDim; ++Dim) {
        LaunchInfo.LocalWorkSize[Dim] = 1;
      }
    }
  }

  // Prepare launch info data
  LaunchInfo.Data.Host.GlobalShadowOffset = DI->Shadow->ShadowBegin;
  LaunchInfo.Data.Host.GlobalShadowOffsetEnd = DI->Shadow->ShadowEnd;
  LaunchInfo.Data.Host.DeviceTy = DI->Type;
  LaunchInfo.Data.Host.Debug = getContext()->Options.Debug ? 1 : 0;

  const size_t *LocalWorkSize = LaunchInfo.LocalWorkSize.data();
  uint32_t NumWG = 1;
  for (uint32_t Dim = 0; Dim < LaunchInfo.WorkDim; ++Dim) {
    NumWG *= (LaunchInfo.GlobalWorkSize[Dim] + LocalWorkSize[Dim] - 1) /
             LocalWorkSize[Dim];
  }

  if (DI->Shadow->AllocLocalShadow(
          Queue, NumWG, LaunchInfo.Data.Host.LocalShadowOffset,
          LaunchInfo.Data.Host.LocalShadowOffsetEnd) != UR_RESULT_SUCCESS) {
    UR_LOG_L(getContext()->logger, WARN,
             "Failed to allocate shadow memory for local memory, "
             "maybe the number of workgroup ({}) is too large",
             NumWG);
    UR_LOG_L(getContext()->logger, WARN,
             "Skip checking local memory of kernel <{}> ",
             GetKernelName(Kernel));
  } else {
    UR_LOG_L(getContext()->logger, DEBUG,
             "ShadowMemory(Local, WorkGroup={}, {} - {})", NumWG,
             (void *)LaunchInfo.Data.Host.LocalShadowOffset,
             (void *)LaunchInfo.Data.Host.LocalShadowOffsetEnd);

    // Write local arguments info
    if (!KernelInfo.LocalArgs.empty()) {
      std::vector<TsanLocalArgsInfo> LocalArgsInfo;
      for (auto [ArgIndex, ArgInfo] : KernelInfo.LocalArgs) {
        LocalArgsInfo.push_back(ArgInfo);
        UR_LOG_L(getContext()->logger, DEBUG,
                 "LocalArgs (argIndex={}, size={})", ArgIndex, ArgInfo.Size);
      }
      UR_CALL(LaunchInfo.Data.importLocalArgsInfo(Queue, LocalArgsInfo));
    }
  }

  LaunchInfo.Data.syncToDevice(Queue);

  // EnqueueWrite __TsanLaunchInfo
  void *LaunchInfoPtr = LaunchInfo.Data.getDevicePtr();
  ur_result_t URes =
      getContext()->urDdiTable.Enqueue.pfnDeviceGlobalVariableWrite(
          Queue, GetProgram(Kernel), "__TsanLaunchInfo", true,
          sizeof(LaunchInfoPtr), 0, &LaunchInfoPtr, 0, nullptr, nullptr);
  if (URes != UR_RESULT_SUCCESS) {
    UR_LOG_L(getContext()->logger, INFO,
             "EnqueueWriteGlobal(__TsanLaunchInfo) "
             "failed, maybe empty kernel: {}",
             URes);
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t
TsanInterceptor::updateShadowMemory(std::shared_ptr<ContextInfo> &CI,
                                    std::shared_ptr<DeviceInfo> &DI,
                                    ur_queue_handle_t Queue) {
  std::scoped_lock<ur_shared_mutex> Guard(CI->AllocInfosMapMutex);
  for (auto &AllocInfo : CI->AllocInfosMap[DI->Handle]) {
    UR_CALL(DI->Shadow->CleanShadow(Queue, AllocInfo.AllocBegin,
                                    AllocInfo.AllocSize));
  }
  CI->AllocInfosMap[DI->Handle].clear();
  return UR_RESULT_SUCCESS;
}

} // namespace tsan

using namespace tsan;

static TsanInterceptor *interceptor;

TsanInterceptor *getTsanInterceptor() { return interceptor; }

void initTsanInterceptor() {
  if (interceptor) {
    return;
  }
  interceptor = new TsanInterceptor();
}

void destroyTsanInterceptor() {
  delete interceptor;
  interceptor = nullptr;
}

} // namespace ur_sanitizer_layer
