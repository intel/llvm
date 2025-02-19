//===----------------------------------------------------------------------===//
/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_interceptor.cpp
 *
 */

#include "asan_interceptor.hpp"
#include "asan_ddi.hpp"
#include "asan_options.hpp"
#include "asan_quarantine.hpp"
#include "asan_report.hpp"
#include "asan_shadow.hpp"
#include "asan_validator.hpp"
#include "sanitizer_common/sanitizer_stacktrace.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"

namespace ur_sanitizer_layer {
namespace asan {

AsanInterceptor::AsanInterceptor() {
  if (getOptions().MaxQuarantineSizeMB) {
    m_Quarantine = std::make_unique<Quarantine>(
        static_cast<uint64_t>(getOptions().MaxQuarantineSizeMB) * 1024 * 1024);
  }
}

AsanInterceptor::~AsanInterceptor() {
  // We must release these objects before releasing adapters, since
  // they may use the adapter in their destructor
  for (const auto &[_, DeviceInfo] : m_DeviceMap) {
    DeviceInfo->Shadow = nullptr;
  }

  m_Quarantine = nullptr;
  m_MemBufferMap.clear();
  m_KernelMap.clear();
  m_ContextMap.clear();
  // AllocationMap need to be cleared after ContextMap because memory leak
  // detection depends on it.
  m_AllocationMap.clear();

  for (auto &[_, ShadowMemory] : m_ShadowMap) {
    ShadowMemory->Destory();
    getContext()->urDdiTable.Context.pfnRelease(ShadowMemory->Context);
  }

  for (auto Adapter : m_Adapters) {
    getContext()->urDdiTable.Global.pfnAdapterRelease(Adapter);
  }
}

/// The memory chunk allocated from the underlying allocator looks like this:
/// L L L L L L U U U U U U R R
///   L -- left redzone words (0 or more bytes)
///   U -- user memory.
///   R -- right redzone (0 or more bytes)
///
/// ref: "compiler-rt/lib/asan/asan_allocator.cpp" Allocator::Allocate
ur_result_t AsanInterceptor::allocateMemory(ur_context_handle_t Context,
                                            ur_device_handle_t Device,
                                            const ur_usm_desc_t *Properties,
                                            ur_usm_pool_handle_t Pool,
                                            size_t Size, AllocType Type,
                                            void **ResultPtr) {

  auto ContextInfo = getContextInfo(Context);
  std::shared_ptr<DeviceInfo> DeviceInfo =
      Device ? getDeviceInfo(Device) : nullptr;

  /// Modified from llvm/compiler-rt/lib/asan/asan_allocator.cpp
  uint32_t Alignment = Properties ? Properties->align : 0;
  // Alignment must be zero or a power-of-two
  if (0 != (Alignment & (Alignment - 1))) {
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  }

  const uint32_t MinAlignment = ASAN_SHADOW_GRANULARITY;
  if (Alignment == 0) {
    Alignment = DeviceInfo ? DeviceInfo->Alignment : MinAlignment;
  }
  if (Alignment < MinAlignment) {
    Alignment = MinAlignment;
  }

  uptr RZLog =
      ComputeRZLog(Size, getOptions().MinRZSize, getOptions().MaxRZSize);
  uptr RZSize = RZLog2Size(RZLog);
  uptr RoundedSize = RoundUpTo(Size, Alignment);
  uptr NeededSize = RoundedSize + RZSize * 2;
  if (Alignment > MinAlignment) {
    NeededSize += Alignment;
  }

  void *Allocated = nullptr;

  if (Pool == nullptr) {
    Pool = ContextInfo->getUSMPool();
  }

  if (Type == AllocType::DEVICE_USM) {
    UR_CALL(getContext()->urDdiTable.USM.pfnDeviceAlloc(
        Context, Device, Properties, Pool, NeededSize, &Allocated));
  } else if (Type == AllocType::HOST_USM) {
    UR_CALL(getContext()->urDdiTable.USM.pfnHostAlloc(Context, Properties, Pool,
                                                      NeededSize, &Allocated));
  } else if (Type == AllocType::SHARED_USM) {
    UR_CALL(getContext()->urDdiTable.USM.pfnSharedAlloc(
        Context, Device, Properties, Pool, NeededSize, &Allocated));
  } else if (Type == AllocType::MEM_BUFFER) {
    UR_CALL(getContext()->urDdiTable.USM.pfnDeviceAlloc(
        Context, Device, Properties, Pool, NeededSize, &Allocated));
  } else {
    getContext()->logger.error("Unsupport memory type");
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  }

  // Udpate statistics
  ContextInfo->Stats.UpdateUSMMalloced(NeededSize, NeededSize - Size);

  uptr AllocBegin = reinterpret_cast<uptr>(Allocated);
  [[maybe_unused]] uptr AllocEnd = AllocBegin + NeededSize;
  uptr UserBegin = AllocBegin + RZSize;
  if (!IsAligned(UserBegin, Alignment)) {
    UserBegin = RoundUpTo(UserBegin, Alignment);
  }
  uptr UserEnd = UserBegin + Size;
  assert(UserEnd <= AllocEnd);

  *ResultPtr = reinterpret_cast<void *>(UserBegin);

  auto AI = std::make_shared<AllocInfo>(AllocInfo{AllocBegin,
                                                  UserBegin,
                                                  UserEnd,
                                                  NeededSize,
                                                  Type,
                                                  false,
                                                  Context,
                                                  Device,
                                                  GetCurrentBacktrace(),
                                                  {}});

  AI->print();

  // For updating shadow memory
  if (Device) { // Device/Shared USM
    ContextInfo->insertAllocInfo({Device}, AI);
  } else { // Host USM
    ContextInfo->insertAllocInfo(ContextInfo->DeviceList, AI);
  }

  // For memory release
  {
    std::scoped_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
    m_AllocationMap.emplace(AI->AllocBegin, std::move(AI));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t AsanInterceptor::releaseMemory(ur_context_handle_t Context,
                                           void *Ptr) {
  auto ContextInfo = getContextInfo(Context);

  auto Addr = reinterpret_cast<uptr>(Ptr);
  auto AllocInfoItOp = findAllocInfoByAddress(Addr);

  if (!AllocInfoItOp) {
    // "Addr" might be a host pointer
    ReportBadFree(Addr, GetCurrentBacktrace(), nullptr);
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  }

  auto AllocInfoIt = *AllocInfoItOp;
  // NOTE: AllocInfoIt will be erased later, so "AllocInfo" must be a new
  // reference here
  auto AllocInfo = AllocInfoIt->second;

  if (AllocInfo->Context != Context) {
    if (AllocInfo->UserBegin == Addr) {
      ReportBadContext(Addr, GetCurrentBacktrace(), AllocInfo);
    } else {
      // "Addr" might be a host pointer
      ReportBadFree(Addr, GetCurrentBacktrace(), nullptr);
    }
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  }

  if (Addr != AllocInfo->UserBegin) {
    ReportBadFree(Addr, GetCurrentBacktrace(), AllocInfo);
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  }

  if (AllocInfo->IsReleased) {
    ReportDoubleFree(Addr, GetCurrentBacktrace(), AllocInfo);
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  }

  AllocInfo->IsReleased = true;
  AllocInfo->ReleaseStack = GetCurrentBacktrace();

  if (AllocInfo->Type == AllocType::HOST_USM) {
    ContextInfo->insertAllocInfo(ContextInfo->DeviceList, AllocInfo);
  } else {
    ContextInfo->insertAllocInfo({AllocInfo->Device}, AllocInfo);
  }

  // If quarantine is disabled, USM is freed immediately
  if (!m_Quarantine) {
    getContext()->logger.debug("Free: {}", (void *)AllocInfo->AllocBegin);

    ContextInfo->Stats.UpdateUSMRealFreed(AllocInfo->AllocSize,
                                          AllocInfo->getRedzoneSize());

    std::scoped_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
    m_AllocationMap.erase(AllocInfoIt);

    return getContext()->urDdiTable.USM.pfnFree(
        Context, (void *)(AllocInfo->AllocBegin));
  }

  // If quarantine is enabled, cache it
  auto ReleaseList = m_Quarantine->put(AllocInfo->Device, AllocInfoIt);
  if (ReleaseList.size()) {
    std::scoped_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
    for (auto &It : ReleaseList) {
      auto ToFreeAllocInfo = It->second;
      getContext()->logger.info("Quarantine Free: {}",
                                (void *)ToFreeAllocInfo->AllocBegin);

      ContextInfo->Stats.UpdateUSMRealFreed(ToFreeAllocInfo->AllocSize,
                                            ToFreeAllocInfo->getRedzoneSize());

      UR_CALL(getContext()->urDdiTable.USM.pfnFree(
          Context, (void *)(ToFreeAllocInfo->AllocBegin)));

      // Erase it at last to avoid use-after-free.
      m_AllocationMap.erase(It);
    }
  }
  ContextInfo->Stats.UpdateUSMFreed(AllocInfo->AllocSize);

  return UR_RESULT_SUCCESS;
}

ur_result_t AsanInterceptor::preLaunchKernel(ur_kernel_handle_t Kernel,
                                             ur_queue_handle_t Queue,
                                             LaunchInfo &LaunchInfo) {
  auto Context = GetContext(Queue);
  auto Device = GetDevice(Queue);
  auto ContextInfo = getContextInfo(Context);
  auto DeviceInfo = getDeviceInfo(Device);

  ManagedQueue InternalQueue(Context, Device);
  if (!InternalQueue) {
    getContext()->logger.error("Failed to create internal queue");
    return UR_RESULT_ERROR_INVALID_QUEUE;
  }

  UR_CALL(prepareLaunch(ContextInfo, DeviceInfo, InternalQueue, Kernel,
                        LaunchInfo));

  UR_CALL(updateShadowMemory(ContextInfo, DeviceInfo, InternalQueue));

  return UR_RESULT_SUCCESS;
}

ur_result_t AsanInterceptor::postLaunchKernel(ur_kernel_handle_t Kernel,
                                              ur_queue_handle_t Queue,
                                              LaunchInfo &LaunchInfo) {
  // FIXME: We must use block operation here, until we support
  // urEventSetCallback
  auto Result = getContext()->urDdiTable.Queue.pfnFinish(Queue);

  UR_CALL(LaunchInfo.Data.syncFromDevice(Queue));

  if (Result == UR_RESULT_SUCCESS) {
    for (const auto &Report : LaunchInfo.Data.Host.Report) {
      if (!Report.Flag) {
        continue;
      }
      switch (Report.ErrorTy) {
      case ErrorType::USE_AFTER_FREE:
        ReportUseAfterFree(Report, Kernel, GetContext(Queue));
        break;
      case ErrorType::OUT_OF_BOUNDS:
      case ErrorType::MISALIGNED:
      case ErrorType::NULL_POINTER:
        ReportGenericError(Report, Kernel);
        break;
      default:
        ReportFatalError(Report);
      }
      if (!Report.IsRecover) {
        exitWithErrors();
      }
    }
  }

  return Result;
}

std::shared_ptr<ShadowMemory>
AsanInterceptor::getOrCreateShadowMemory(ur_device_handle_t Device,
                                         DeviceType Type) {
  std::scoped_lock<ur_shared_mutex> Guard(m_ShadowMapMutex);
  if (m_ShadowMap.find(Type) == m_ShadowMap.end()) {
    ur_context_handle_t InternalContext;
    auto Res = getContext()->urDdiTable.Context.pfnCreate(1, &Device, nullptr,
                                                          &InternalContext);
    if (Res != UR_RESULT_SUCCESS) {
      getContext()->logger.error("Failed to create shadow context");
      return nullptr;
    }
    std::shared_ptr<ContextInfo> CI;
    insertContext(InternalContext, CI);
    m_ShadowMap[Type] = GetShadowMemory(InternalContext, Device, Type);
    m_ShadowMap[Type]->Setup();
  }
  return m_ShadowMap[Type];
}

/// Each 8 bytes of application memory are mapped into one byte of shadow memory
/// The meaning of that byte:
///  - Negative: All bytes are not accessible (poisoned)
///  - 0: All bytes are accessible
///  - 1 <= k <= 7: Only the first k bytes is accessible
///
/// ref:
/// https://github.com/google/sanitizers/wiki/AddressSanitizerAlgorithm#mapping
ur_result_t
AsanInterceptor::enqueueAllocInfo(std::shared_ptr<DeviceInfo> &DeviceInfo,
                                  ur_queue_handle_t Queue,
                                  std::shared_ptr<AllocInfo> &AI) {
  if (AI->IsReleased) {
    int ShadowByte;
    switch (AI->Type) {
    case AllocType::HOST_USM:
      ShadowByte = kUsmHostDeallocatedMagic;
      break;
    case AllocType::DEVICE_USM:
      ShadowByte = kUsmDeviceDeallocatedMagic;
      break;
    case AllocType::SHARED_USM:
      ShadowByte = kUsmSharedDeallocatedMagic;
      break;
    case AllocType::MEM_BUFFER:
      ShadowByte = kMemBufferDeallocatedMagic;
      break;
    default:
      ShadowByte = 0xff;
      assert(false && "Unknow AllocInfo Type");
    }
    UR_CALL(DeviceInfo->Shadow->EnqueuePoisonShadow(Queue, AI->AllocBegin,
                                                    AI->AllocSize, ShadowByte));
    return UR_RESULT_SUCCESS;
  }

  // Init zero
  UR_CALL(DeviceInfo->Shadow->EnqueuePoisonShadow(Queue, AI->AllocBegin,
                                                  AI->AllocSize, 0));

  uptr TailBegin = RoundUpTo(AI->UserEnd, ASAN_SHADOW_GRANULARITY);
  uptr TailEnd = AI->AllocBegin + AI->AllocSize;

  // User tail
  if (TailBegin != AI->UserEnd) {
    auto Value =
        AI->UserEnd - RoundDownTo(AI->UserEnd, ASAN_SHADOW_GRANULARITY);
    UR_CALL(DeviceInfo->Shadow->EnqueuePoisonShadow(Queue, AI->UserEnd, 1,
                                                    static_cast<u8>(Value)));
  }

  int ShadowByte;
  switch (AI->Type) {
  case AllocType::HOST_USM:
    ShadowByte = kUsmHostRedzoneMagic;
    break;
  case AllocType::DEVICE_USM:
    ShadowByte = kUsmDeviceRedzoneMagic;
    break;
  case AllocType::SHARED_USM:
    ShadowByte = kUsmSharedRedzoneMagic;
    break;
  case AllocType::MEM_BUFFER:
    ShadowByte = kMemBufferRedzoneMagic;
    break;
  case AllocType::DEVICE_GLOBAL:
    ShadowByte = kDeviceGlobalRedzoneMagic;
    break;
  default:
    ShadowByte = 0xff;
    assert(false && "Unknow AllocInfo Type");
  }

  // Left red zone
  UR_CALL(DeviceInfo->Shadow->EnqueuePoisonShadow(
      Queue, AI->AllocBegin, AI->UserBegin - AI->AllocBegin, ShadowByte));

  // Right red zone
  UR_CALL(DeviceInfo->Shadow->EnqueuePoisonShadow(
      Queue, TailBegin, TailEnd - TailBegin, ShadowByte));

  return UR_RESULT_SUCCESS;
}

ur_result_t
AsanInterceptor::updateShadowMemory(std::shared_ptr<ContextInfo> &ContextInfo,
                                    std::shared_ptr<DeviceInfo> &DeviceInfo,
                                    ur_queue_handle_t Queue) {
  auto &AllocInfos = ContextInfo->AllocInfosMap[DeviceInfo->Handle];
  std::scoped_lock<ur_shared_mutex> Guard(AllocInfos.Mutex);

  for (auto &AI : AllocInfos.List) {
    UR_CALL(enqueueAllocInfo(DeviceInfo, Queue, AI));
  }
  AllocInfos.List.clear();

  return UR_RESULT_SUCCESS;
}

ur_result_t AsanInterceptor::registerProgram(ur_program_handle_t Program) {
  ur_result_t Result = UR_RESULT_SUCCESS;

  getContext()->logger.info("registerSpirKernels");
  Result = registerSpirKernels(Program);
  if (Result != UR_RESULT_SUCCESS) {
    return Result;
  }

  getContext()->logger.info("registerDeviceGlobals");
  Result = registerDeviceGlobals(Program);
  if (Result != UR_RESULT_SUCCESS) {
    return Result;
  }

  return Result;
}

ur_result_t AsanInterceptor::unregisterProgram(ur_program_handle_t Program) {
  auto ProgramInfo = getProgramInfo(Program);
  assert(ProgramInfo != nullptr && "unregistered program!");

  std::scoped_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
  for (auto AI : ProgramInfo->AllocInfoForGlobals) {
    m_AllocationMap.erase(AI->AllocBegin);
  }
  ProgramInfo->AllocInfoForGlobals.clear();

  ProgramInfo->InstrumentedKernels.clear();

  return UR_RESULT_SUCCESS;
}

ur_result_t AsanInterceptor::registerSpirKernels(ur_program_handle_t Program) {
  auto Context = GetContext(Program);
  std::vector<ur_device_handle_t> Devices = GetDevices(Program);

  for (auto Device : Devices) {
    size_t MetadataSize;
    void *MetadataPtr;
    ur_result_t Result =
        getContext()->urDdiTable.Program.pfnGetGlobalVariablePointer(
            Device, Program, kSPIR_AsanSpirKernelMetadata, &MetadataSize,
            &MetadataPtr);
    if (Result != UR_RESULT_SUCCESS) {
      continue;
    }

    const uint64_t NumOfSpirKernel = MetadataSize / sizeof(SpirKernelInfo);
    assert((MetadataSize % sizeof(SpirKernelInfo) == 0) &&
           "SpirKernelMetadata size is not correct");

    ManagedQueue Queue(Context, Device);

    std::vector<SpirKernelInfo> SKInfo(NumOfSpirKernel);
    Result = getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
        Queue, true, &SKInfo[0], MetadataPtr,
        sizeof(SpirKernelInfo) * NumOfSpirKernel, 0, nullptr, nullptr);
    if (Result != UR_RESULT_SUCCESS) {
      getContext()->logger.error("Can't read the value of <{}>: {}",
                                 kSPIR_AsanSpirKernelMetadata, Result);
      return Result;
    }

    auto PI = getProgramInfo(Program);
    assert(PI != nullptr && "unregistered program!");
    for (const auto &SKI : SKInfo) {
      if (SKI.Size == 0) {
        continue;
      }
      std::vector<char> KernelNameV(SKI.Size);
      Result = getContext()->urDdiTable.Enqueue.pfnUSMMemcpy(
          Queue, true, KernelNameV.data(), (void *)SKI.KernelName,
          sizeof(char) * SKI.Size, 0, nullptr, nullptr);
      if (Result != UR_RESULT_SUCCESS) {
        getContext()->logger.error("Can't read kernel name: {}", Result);
        return Result;
      }

      std::string KernelName =
          std::string(KernelNameV.begin(), KernelNameV.end());

      getContext()->logger.info("SpirKernel(name='{}', isInstrumented={})",
                                KernelName, true);

      PI->InstrumentedKernels.insert(std::move(KernelName));
    }
    getContext()->logger.info("Number of sanitized kernel: {}",
                              PI->InstrumentedKernels.size());
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t
AsanInterceptor::registerDeviceGlobals(ur_program_handle_t Program) {
  std::vector<ur_device_handle_t> Devices = GetDevices(Program);
  assert(Devices.size() != 0 && "No devices in registerDeviceGlobals");
  auto Context = GetContext(Program);
  auto ContextInfo = getContextInfo(Context);
  auto ProgramInfo = getProgramInfo(Program);
  assert(ProgramInfo != nullptr && "unregistered program!");

  for (auto Device : Devices) {
    ManagedQueue Queue(Context, Device);

    size_t MetadataSize;
    void *MetadataPtr;
    auto Result = getContext()->urDdiTable.Program.pfnGetGlobalVariablePointer(
        Device, Program, kSPIR_AsanDeviceGlobalMetadata, &MetadataSize,
        &MetadataPtr);
    if (Result != UR_RESULT_SUCCESS) {
      getContext()->logger.info("No device globals");
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
      getContext()->logger.error("Device Global[{}] Read Failed: {}",
                                 kSPIR_AsanDeviceGlobalMetadata, Result);
      return Result;
    }

    for (size_t i = 0; i < NumOfDeviceGlobal; i++) {
      auto AI = std::make_shared<AllocInfo>(
          AllocInfo{GVInfos[i].Addr,
                    GVInfos[i].Addr,
                    GVInfos[i].Addr + GVInfos[i].Size,
                    GVInfos[i].SizeWithRedZone,
                    AllocType::DEVICE_GLOBAL,
                    false,
                    Context,
                    Device,
                    GetCurrentBacktrace(),
                    {}});

      ContextInfo->insertAllocInfo({Device}, AI);
      ProgramInfo->AllocInfoForGlobals.emplace(AI);

      std::scoped_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
      m_AllocationMap.emplace(AI->AllocBegin, std::move(AI));
    }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t AsanInterceptor::insertContext(ur_context_handle_t Context,
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

ur_result_t AsanInterceptor::eraseContext(ur_context_handle_t Context) {
  std::scoped_lock<ur_shared_mutex> Guard(m_ContextMapMutex);
  assert(m_ContextMap.find(Context) != m_ContextMap.end());
  m_ContextMap.erase(Context);
  // TODO: Remove devices in each context
  return UR_RESULT_SUCCESS;
}

ur_result_t AsanInterceptor::insertDevice(ur_device_handle_t Device,
                                          std::shared_ptr<DeviceInfo> &DI) {
  std::scoped_lock<ur_shared_mutex> Guard(m_DeviceMapMutex);

  if (m_DeviceMap.find(Device) != m_DeviceMap.end()) {
    DI = m_DeviceMap.at(Device);
    return UR_RESULT_SUCCESS;
  }

  DI = std::make_shared<DeviceInfo>(Device);

  DI->IsSupportSharedSystemUSM =
      GetDeviceUSMCapability(Device, UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT);

  // Query alignment
  UR_CALL(getContext()->urDdiTable.Device.pfnGetInfo(
      Device, UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN, sizeof(DI->Alignment),
      &DI->Alignment, nullptr));

  // Don't move DI, since it's a return value as well
  m_DeviceMap.emplace(Device, DI);

  return UR_RESULT_SUCCESS;
}

ur_result_t AsanInterceptor::eraseDevice(ur_device_handle_t Device) {
  std::scoped_lock<ur_shared_mutex> Guard(m_DeviceMapMutex);
  assert(m_DeviceMap.find(Device) != m_DeviceMap.end());
  m_DeviceMap.erase(Device);
  // TODO: Remove devices in each context
  return UR_RESULT_SUCCESS;
}

ur_result_t AsanInterceptor::insertProgram(ur_program_handle_t Program) {
  std::scoped_lock<ur_shared_mutex> Guard(m_ProgramMapMutex);
  if (m_ProgramMap.find(Program) != m_ProgramMap.end()) {
    return UR_RESULT_SUCCESS;
  }
  m_ProgramMap.emplace(Program, std::make_shared<ProgramInfo>(Program));
  return UR_RESULT_SUCCESS;
}

ur_result_t AsanInterceptor::eraseProgram(ur_program_handle_t Program) {
  std::scoped_lock<ur_shared_mutex> Guard(m_ProgramMapMutex);
  assert(m_ProgramMap.find(Program) != m_ProgramMap.end());
  m_ProgramMap.erase(Program);
  return UR_RESULT_SUCCESS;
}

KernelInfo &AsanInterceptor::getOrCreateKernelInfo(ur_kernel_handle_t Kernel) {
  {
    std::shared_lock<ur_shared_mutex> Guard(m_KernelMapMutex);
    if (m_KernelMap.find(Kernel) != m_KernelMap.end()) {
      return *m_KernelMap[Kernel].get();
    }
  }

  // Create new KernelInfo
  auto Program = GetProgram(Kernel);
  auto PI = getProgramInfo(Program);
  bool IsInstrumented = PI->isKernelInstrumented(Kernel);

  std::scoped_lock<ur_shared_mutex> Guard(m_KernelMapMutex);
  m_KernelMap.emplace(Kernel,
                      std::make_unique<KernelInfo>(Kernel, IsInstrumented));
  return *m_KernelMap[Kernel].get();
}

ur_result_t AsanInterceptor::eraseKernelInfo(ur_kernel_handle_t Kernel) {
  std::scoped_lock<ur_shared_mutex> Guard(m_KernelMapMutex);
  assert(m_KernelMap.find(Kernel) != m_KernelMap.end());
  m_KernelMap.erase(Kernel);
  return UR_RESULT_SUCCESS;
}

ur_result_t
AsanInterceptor::insertMemBuffer(std::shared_ptr<MemBuffer> MemBuffer) {
  std::scoped_lock<ur_shared_mutex> Guard(m_MemBufferMapMutex);
  assert(m_MemBufferMap.find(ur_cast<ur_mem_handle_t>(MemBuffer.get())) ==
         m_MemBufferMap.end());
  m_MemBufferMap.emplace(reinterpret_cast<ur_mem_handle_t>(MemBuffer.get()),
                         MemBuffer);
  return UR_RESULT_SUCCESS;
}

ur_result_t AsanInterceptor::eraseMemBuffer(ur_mem_handle_t MemHandle) {
  std::scoped_lock<ur_shared_mutex> Guard(m_MemBufferMapMutex);
  assert(m_MemBufferMap.find(MemHandle) != m_MemBufferMap.end());
  m_MemBufferMap.erase(MemHandle);
  return UR_RESULT_SUCCESS;
}

std::shared_ptr<MemBuffer>
AsanInterceptor::getMemBuffer(ur_mem_handle_t MemHandle) {
  std::shared_lock<ur_shared_mutex> Guard(m_MemBufferMapMutex);
  if (m_MemBufferMap.find(MemHandle) != m_MemBufferMap.end()) {
    return m_MemBufferMap[MemHandle];
  }
  return nullptr;
}

ur_result_t AsanInterceptor::prepareLaunch(
    std::shared_ptr<ContextInfo> &ContextInfo,
    std::shared_ptr<DeviceInfo> &DeviceInfo, ur_queue_handle_t Queue,
    ur_kernel_handle_t Kernel, LaunchInfo &LaunchInfo) {
  auto &KernelInfo = getOrCreateKernelInfo(Kernel);
  std::shared_lock<ur_shared_mutex> Guard(KernelInfo.Mutex);

  auto ArgNums = GetKernelNumArgs(Kernel);
  auto LocalMemoryUsage = GetKernelLocalMemorySize(Kernel, DeviceInfo->Handle);
  auto PrivateMemoryUsage =
      GetKernelPrivateMemorySize(Kernel, DeviceInfo->Handle);

  getContext()->logger.info(
      "KernelInfo {} (Name={}, ArgNums={}, IsInstrumented={}, "
      "LocalMemory={}, PrivateMemory={})",
      (void *)Kernel, GetKernelName(Kernel), ArgNums, KernelInfo.IsInstrumented,
      LocalMemoryUsage, PrivateMemoryUsage);

  // Validate pointer arguments
  if (getOptions().DetectKernelArguments) {
    for (const auto &[ArgIndex, PtrPair] : KernelInfo.PointerArgs) {
      auto Ptr = PtrPair.first;
      if (Ptr == nullptr) {
        continue;
      }
      if (auto ValidateResult = ValidateUSMPointer(
              ContextInfo->Handle, DeviceInfo->Handle, (uptr)Ptr)) {
        ReportInvalidKernelArgument(Kernel, ArgIndex, (uptr)Ptr, ValidateResult,
                                    PtrPair.second);
        if (ValidateResult.Type != ValidateUSMResult::MAYBE_HOST_POINTER) {
          exitWithErrors();
        }
      }
    }
  }

  // Set membuffer arguments
  for (const auto &[ArgIndex, MemBuffer] : KernelInfo.BufferArgs) {
    char *ArgPointer = nullptr;
    UR_CALL(MemBuffer->getHandle(DeviceInfo->Handle, ArgPointer));
    ur_result_t URes = getContext()->urDdiTable.Kernel.pfnSetArgPointer(
        Kernel, ArgIndex, nullptr, ArgPointer);
    if (URes != UR_RESULT_SUCCESS) {
      getContext()->logger.error(
          "Failed to set buffer {} as the {} arg to kernel {}: {}",
          ur_cast<ur_mem_handle_t>(MemBuffer.get()), ArgIndex, Kernel, URes);
    }
  }

  if (!KernelInfo.IsInstrumented) {
    return UR_RESULT_SUCCESS;
  }

  // We must prepare all kernel args before call
  // urKernelGetSuggestedLocalWorkSize, otherwise the call will fail on
  // CPU device.
  {
    assert(ArgNums >= 1 &&
           "Sanitized Kernel should have at least one argument");

    ur_result_t URes = getContext()->urDdiTable.Kernel.pfnSetArgPointer(
        Kernel, ArgNums - 1, nullptr, LaunchInfo.Data.getDevicePtr());
    if (URes != UR_RESULT_SUCCESS) {
      getContext()->logger.error("Failed to set launch info: {}", URes);
      return URes;
    }
  }

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

  const size_t *LocalWorkSize = LaunchInfo.LocalWorkSize.data();
  uint32_t NumWG = 1;
  for (uint32_t Dim = 0; Dim < LaunchInfo.WorkDim; ++Dim) {
    NumWG *= (LaunchInfo.GlobalWorkSize[Dim] + LocalWorkSize[Dim] - 1) /
             LocalWorkSize[Dim];
  }

  // Prepare asan runtime data
  LaunchInfo.Data.Host.GlobalShadowOffset = DeviceInfo->Shadow->ShadowBegin;
  LaunchInfo.Data.Host.GlobalShadowOffsetEnd = DeviceInfo->Shadow->ShadowEnd;
  LaunchInfo.Data.Host.DeviceTy = DeviceInfo->Type;
  LaunchInfo.Data.Host.Debug = getOptions().Debug ? 1 : 0;

  // Write shadow memory offset for local memory
  if (getOptions().DetectLocals) {
    if (DeviceInfo->Shadow->AllocLocalShadow(
            Queue, NumWG, LaunchInfo.Data.Host.LocalShadowOffset,
            LaunchInfo.Data.Host.LocalShadowOffsetEnd) != UR_RESULT_SUCCESS) {
      getContext()->logger.warning(
          "Failed to allocate shadow memory for local "
          "memory, maybe the number of workgroup ({}) is too "
          "large",
          NumWG);
      getContext()->logger.warning("Skip checking local memory of kernel <{}>",
                                   GetKernelName(Kernel));
    } else {
      getContext()->logger.info(
          "ShadowMemory(Local, WorkGroup{}, {} - {})", NumWG,
          (void *)LaunchInfo.Data.Host.LocalShadowOffset,
          (void *)LaunchInfo.Data.Host.LocalShadowOffsetEnd);
    }
  }

  // Write shadow memory offset for private memory
  if (getOptions().DetectPrivates) {
    if (DeviceInfo->Shadow->AllocPrivateShadow(
            Queue, NumWG, LaunchInfo.Data.Host.PrivateShadowOffset,
            LaunchInfo.Data.Host.PrivateShadowOffsetEnd) != UR_RESULT_SUCCESS) {
      getContext()->logger.warning(
          "Failed to allocate shadow memory for private "
          "memory, maybe the number of workgroup ({}) is too "
          "large",
          NumWG);
      getContext()->logger.warning(
          "Skip checking private memory of kernel <{}>", GetKernelName(Kernel));
    } else {
      getContext()->logger.info(
          "ShadowMemory(Private, WorkGroup{}, {} - {})", NumWG,
          (void *)LaunchInfo.Data.Host.PrivateShadowOffset,
          (void *)LaunchInfo.Data.Host.PrivateShadowOffsetEnd);
    }
  }

  // Write local arguments info
  if (!KernelInfo.LocalArgs.empty()) {
    std::vector<LocalArgsInfo> LocalArgsInfo;
    for (auto [ArgIndex, ArgInfo] : KernelInfo.LocalArgs) {
      LocalArgsInfo.push_back(ArgInfo);
      getContext()->logger.debug(
          "local_args (argIndex={}, size={}, sizeWithRZ={})", ArgIndex,
          ArgInfo.Size, ArgInfo.SizeWithRedZone);
    }
    UR_CALL(LaunchInfo.Data.importLocalArgsInfo(Queue, LocalArgsInfo));
  }

  // sync asan runtime data to device side
  UR_CALL(LaunchInfo.Data.syncToDevice(Queue));

  getContext()->logger.info(
      "LaunchInfo {} (device={}, debug={}, numLocalArgs={}, localArgs={})",
      (void *)LaunchInfo.Data.getDevicePtr(),
      ToString(LaunchInfo.Data.Host.DeviceTy), LaunchInfo.Data.Host.Debug,
      LaunchInfo.Data.Host.NumLocalArgs,
      (void *)LaunchInfo.Data.Host.LocalArgs);

  return UR_RESULT_SUCCESS;
}

std::optional<AllocationIterator>
AsanInterceptor::findAllocInfoByAddress(uptr Address) {
  std::shared_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
  auto It = m_AllocationMap.upper_bound(Address);
  if (It == m_AllocationMap.begin()) {
    return std::nullopt;
  }
  --It;

  // Maybe it's a host pointer
  if (Address < It->second->AllocBegin ||
      Address >= It->second->AllocBegin + It->second->AllocSize) {
    return std::nullopt;
  }
  return It;
}

std::vector<AllocationIterator>
AsanInterceptor::findAllocInfoByContext(ur_context_handle_t Context) {
  std::shared_lock<ur_shared_mutex> Guard(m_AllocationMapMutex);
  std::vector<AllocationIterator> AllocInfos;
  for (auto It = m_AllocationMap.begin(); It != m_AllocationMap.end(); It++) {
    const auto &[_, AI] = *It;
    if (AI->Context == Context) {
      AllocInfos.emplace_back(It);
    }
  }
  return AllocInfos;
}

bool ProgramInfo::isKernelInstrumented(ur_kernel_handle_t Kernel) const {
  const auto Name = GetKernelName(Kernel);
  return InstrumentedKernels.find(Name) != InstrumentedKernels.end();
}

ContextInfo::~ContextInfo() {
  Stats.Print(Handle);

  [[maybe_unused]] ur_result_t URes;
  if (USMPool) {
    URes = getContext()->urDdiTable.USM.pfnPoolRelease(USMPool);
    assert(URes == UR_RESULT_SUCCESS);
  }

  URes = getContext()->urDdiTable.Context.pfnRelease(Handle);
  assert(URes == UR_RESULT_SUCCESS);

  // check memory leaks
  if (getAsanInterceptor()->getOptions().DetectLeaks &&
      getAsanInterceptor()->isNormalExit()) {
    std::vector<AllocationIterator> AllocInfos =
        getAsanInterceptor()->findAllocInfoByContext(Handle);
    for (const auto &It : AllocInfos) {
      const auto &[_, AI] = *It;
      if (!AI->IsReleased) {
        ReportMemoryLeak(AI);
      }
    }
  }
}

ur_usm_pool_handle_t ContextInfo::getUSMPool() {
  std::call_once(PoolInit, [this]() {
    ur_usm_pool_desc_t Desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr, 0};
    auto URes =
        getContext()->urDdiTable.USM.pfnPoolCreate(Handle, &Desc, &USMPool);
    if (URes != UR_RESULT_SUCCESS &&
        URes != UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      getContext()->logger.warning(
          "Failed to create USM pool, the memory overhead "
          "may increase: {}",
          URes);
    }
  });
  return USMPool;
}

AsanRuntimeDataWrapper::~AsanRuntimeDataWrapper() {
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

LaunchInfo::~LaunchInfo() {
  [[maybe_unused]] ur_result_t Result;
  Result = getContext()->urDdiTable.Context.pfnRelease(Context);
  assert(Result == UR_RESULT_SUCCESS);
  Result = getContext()->urDdiTable.Device.pfnRelease(Device);
  assert(Result == UR_RESULT_SUCCESS);
}

} // namespace asan

using namespace asan;

static AsanInterceptor *interceptor;

AsanInterceptor *getAsanInterceptor() { return interceptor; }

void initAsanInterceptor() {
  if (interceptor) {
    return;
  }
  interceptor = new AsanInterceptor();
}

void destroyAsanInterceptor() {
  delete interceptor;
  interceptor = nullptr;
}

} // namespace ur_sanitizer_layer
