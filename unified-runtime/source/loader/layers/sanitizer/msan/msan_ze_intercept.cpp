/*
 *
 * Copyright (C) 2026 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_ze_intercept.cpp
 *
 * MSan hooks for the Level Zero calls of liboffload's level_zero plugin. They
 * mirror MsanInterceptor (allocateMemory, registerSpirKernels,
 * registerDeviceGlobals, prepareLaunch, postLaunchKernel) on raw L0 handles.
 */

#include "msan_interceptor.hpp"
#include "msan_origin.hpp"
#include "msan_report.hpp"
#include "msan_shadow.hpp"
#include "msan_ze.hpp"
#include "sanitizer_common/sanitizer_stacktrace.hpp"
#include "ur_sanitizer_layer.hpp"

#include <algorithm>
#include <dlfcn.h>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace ur_sanitizer_layer {
namespace msan {

namespace {

// Non-hooked L0 calls, resolved from the real loader so they never reach the
// shim (the layer doesn't link libze_loader).
#define MSAN_ZE_EXTRA(X)                                                       \
  X(zeMemFree)                                                                 \
  X(zeMemGetAllocProperties)                                                   \
  X(zeVirtualMemReserve)                                                       \
  X(zeVirtualMemQueryPageSize)                                                 \
  X(zeVirtualMemMap)                                                           \
  X(zePhysicalMemCreate)                                                       \
  X(zeModuleGetKernelNames)                                                    \
  X(zeModuleGetGlobalPointer)                                                  \
  X(zeKernelGetProperties)                                                     \
  X(zeKernelSetIndirectAccess)                                                 \
  X(zeCommandListCreateImmediate)                                              \
  X(zeCommandListHostSynchronize)                                              \
  X(zeDeviceGetComputeProperties)

struct {
#define MSAN_ZE_FIELD(N) decltype(&::N) N = nullptr;
  MSAN_ZE_EXTRA(MSAN_ZE_FIELD)
#undef MSAN_ZE_FIELD
} Ze;

bool loadZeExtra() {
  void *Handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_NOLOAD);
  if (!Handle) {
    return false;
  }
#define MSAN_ZE_LOAD(N)                                                        \
  if (!(Ze.N = (decltype(Ze.N))dlsym(Handle, #N)))                             \
    return false;
  MSAN_ZE_EXTRA(MSAN_ZE_LOAD)
#undef MSAN_ZE_LOAD
  return true;
}

struct KernelRecord {
  std::string Name;
  ProgramInfo::KernelMetadata Meta;
  void *LaunchInfo; // address of the module's __MsanLaunchInfo
  uint32_t GroupSize[3] = {1, 1, 1};
};

struct ModuleRecord {
  std::unordered_map<std::string, ProgramInfo::KernelMetadata> Kernels;
  void *LaunchInfo = nullptr;
};

// shortcut: single context/device (the first one seen), ordinal 0 for the
// internal list; upgrade for multi-device.
struct ZeState {
  std::mutex Mutex;
  bool Initialized = false;
  bool Enabled = false;
  ze_context_handle_t Context{};
  ze_device_handle_t Device{};
  ze_command_list_handle_t List{}; // internal, synchronous immediate
  MsanShadowMemoryPVC Shadow{nullptr, nullptr}; // address math only
  size_t PageSize = 0;
  size_t LocalMemorySize = 0;
  uint32_t MinSubGroupSize = 1;
  size_t CleanShadowSize = 1024;
  std::unordered_set<uptr> MappedPages;
  std::unordered_map<ze_module_handle_t, ModuleRecord> Modules;
  std::unordered_map<ze_kernel_handle_t, KernelRecord> Kernels;
  // Host allocs made before init() had a device: {Ptr, Size, Value, Origin}
  std::vector<std::tuple<uptr, size_t, u8, uint32_t>> PendingPoison;
};

// Leaked: hooks may still run during process exit.
ZeState &S = *new ZeState;

#define ZE_CALL(Call)                                                          \
  do {                                                                         \
    ze_result_t ZeRes = Call;                                                  \
    if (ZeRes != ZE_RESULT_SUCCESS) {                                          \
      UR_LOG_L(getContext()->logger, ERR, "{}: {}", #Call, (int)ZeRes);        \
      return ZeRes;                                                            \
    }                                                                          \
  } while (0)

// Blocking ops on the internal list.
ze_result_t copy(void *Dst, const void *Src, size_t Size) {
  ZE_CALL(ZeReal->zeCommandListAppendMemoryCopy(S.List, Dst, Src, Size, nullptr,
                                                0, nullptr));
  return Ze.zeCommandListHostSynchronize(S.List, UINT64_MAX);
}

ze_result_t fill(uptr Ptr, const void *Pattern, size_t PatternSize,
                 size_t Size) {
  ZE_CALL(ZeReal->zeCommandListAppendMemoryFill(
      S.List, (void *)Ptr, Pattern, PatternSize, Size, nullptr, 0, nullptr));
  return Ze.zeCommandListHostSynchronize(S.List, UINT64_MAX);
}

void *allocDevice(size_t Size) {
  ze_device_mem_alloc_desc_t Desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  void *Ptr = nullptr;
  if (ZeReal->zeMemAllocDevice(S.Context, &Desc, Size, 0, S.Device, &Ptr) !=
      ZE_RESULT_SUCCESS) {
    return nullptr;
  }
  return Ptr;
}

// Like MsanShadowMemoryGPU::EnqueueVirtualMemMap.
ze_result_t mapPages(uptr Begin, uptr End) {
  for (uptr Page = RoundDownTo(Begin, S.PageSize); Page <= End;
       Page += S.PageSize) {
    if (S.MappedPages.count(Page)) {
      continue;
    }
    ze_physical_mem_desc_t Desc{ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, nullptr, 0,
                                S.PageSize};
    ze_physical_mem_handle_t Phys{};
    ZE_CALL(Ze.zePhysicalMemCreate(S.Context, S.Device, &Desc, &Phys));
    ZE_CALL(Ze.zeVirtualMemMap(S.Context, (void *)Page, S.PageSize, Phys, 0,
                               ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE));
    S.MappedPages.insert(Page);
    UR_LOG_L(getContext()->logger, DEBUG, "zeVirtualMemMap: {} ~ {}",
             (void *)Page, (void *)(Page + S.PageSize - 1));
  }
  return ZE_RESULT_SUCCESS;
}

// Like MsanShadowMemoryGPU::EnqueuePoisonShadowWithOrigin.
ze_result_t poison(uptr Ptr, size_t Size, u8 Value, uint32_t OriginId = 0) {
  if (Size == 0) {
    return ZE_RESULT_SUCCESS;
  }
  uptr ShadowBegin = S.Shadow.MemToShadow(Ptr);
  uptr ShadowEnd = S.Shadow.MemToShadow(Ptr + Size - 1);
  ZE_CALL(mapPages(ShadowBegin, ShadowEnd));
  UR_LOG_L(getContext()->logger, DEBUG,
           "EnqueuePoisonShadow(addr={}, size={}, value={})",
           (void *)ShadowBegin, ShadowEnd - ShadowBegin + 1,
           (void *)(size_t)Value);
  ZE_CALL(fill(ShadowBegin, &Value, 1, ShadowEnd - ShadowBegin + 1));

  uptr OriginBegin = S.Shadow.MemToOrigin(Ptr);
  uptr OriginEnd =
      S.Shadow.MemToOrigin(Ptr + Size - 1) + MSAN_ORIGIN_GRANULARITY - 1;
  ZE_CALL(mapPages(OriginBegin, OriginEnd));
  if (OriginId) {
    ZE_CALL(fill(OriginBegin, &OriginId, sizeof(OriginId),
                 OriginEnd - OriginBegin + 1));
  }
  return ZE_RESULT_SUCCESS;
}

// Must hold S.Mutex. Returns whether MSan is active for this context.
bool init(ze_context_handle_t Context, ze_device_handle_t Device) {
  if (S.Initialized || !Device) {
    return S.Enabled && Context == S.Context;
  }
  S.Initialized = true;
  S.Context = Context;
  S.Device = Device;

  if (!loadZeExtra()) {
    UR_LOG_L(getContext()->logger, ERR,
             "Failed to resolve Level Zero entry points, MSan disabled");
    return false;
  }

  void *Probe = allocDevice(4);
  if (!Probe) {
    return false;
  }
  bool IsPVC = MsanShadowMemoryPVC::IsDeviceUSM((uptr)Probe);
  Ze.zeMemFree(Context, Probe);
  if (!IsPVC) {
    UR_LOG_L(getContext()->logger, WARN,
             "MSan over offload only supports PVC, MSan disabled");
    return false;
  }

  const size_t ShadowSize = S.Shadow.GetShadowSize();
  if (Ze.zeVirtualMemReserve(Context, (void *)S.Shadow.GetStartAddress(),
                             ShadowSize, (void **)&S.Shadow.ShadowBegin) !=
      ZE_RESULT_SUCCESS) {
    UR_LOG_L(getContext()->logger, ERR,
             "Shadow memory reserved failed with size {}", (void *)ShadowSize);
    return false;
  }
  S.Shadow.ShadowEnd = S.Shadow.ShadowBegin + ShadowSize;
  UR_LOG_L(getContext()->logger, INFO, "ShadowMemory(Global): {} - {}",
           (void *)S.Shadow.ShadowBegin, (void *)S.Shadow.ShadowEnd);

  if (Ze.zeVirtualMemQueryPageSize(Context, Device, 1, &S.PageSize) !=
      ZE_RESULT_SUCCESS) {
    return false;
  }

  ze_device_compute_properties_t CP{
      ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES};
  if (Ze.zeDeviceGetComputeProperties(Device, &CP) != ZE_RESULT_SUCCESS) {
    return false;
  }
  S.LocalMemorySize = CP.maxSharedLocalMemory;
  if (CP.numSubGroupSizes) {
    S.MinSubGroupSize = *std::min_element(
        CP.subGroupSizes, CP.subGroupSizes + CP.numSubGroupSizes);
  }

  ze_command_queue_desc_t QD{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  QD.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  if (Ze.zeCommandListCreateImmediate(Context, Device, &QD, &S.List) !=
      ZE_RESULT_SUCCESS) {
    UR_LOG_L(getContext()->logger, ERR, "Failed to create internal list");
    return false;
  }

  S.Enabled = true;
  for (auto [Ptr, Size, Value, OriginId] : S.PendingPoison) {
    poison(Ptr, Size, Value, OriginId);
  }
  S.PendingPoison.clear();
  return true;
}

// Like MsanInterceptor::allocateMemory, after the real allocation.
ze_result_t afterAlloc(ze_result_t Res, ze_context_handle_t Context,
                       ze_device_handle_t Device, void **PtrOut, size_t Size,
                       HeapType Type) {
  if (Res != ZE_RESULT_SUCCESS) {
    return Res;
  }
  void *Ptr = *PtrOut;
  std::scoped_lock<std::mutex> Guard(S.Mutex);
  S.CleanShadowSize = std::max(S.CleanShadowSize, Size);

  bool DontCheckHostOrSharedUSM =
      (Type == HeapType::HostUSM || Type == HeapType::SharedUSM) &&
      !getContext()->Options.MsanCheckHostAndSharedUSM;

  StackTrace Stack = GetCurrentBacktrace();
  Origin HeapOrigin = DontCheckHostOrSharedUSM
                          ? Origin::FromRawId(0)
                          : Origin::CreateHeapOrigin(Stack, Type);

  u8 Value =
      DontCheckHostOrSharedUSM ? kMemInitializedMagic : kMemUninitializedMagic;

  // L0 can't tell us a device for a host alloc's context, so defer it.
  // shortcut: assumes it's the context init() will see.
  if (!S.Initialized && !Device) {
    S.PendingPoison.emplace_back((uptr)Ptr, Size, Value, *HeapOrigin.rawId());
    return Res;
  }
  if (!init(Context, Device)) {
    return Res;
  }

  ze_result_t PoisonRes = poison((uptr)Ptr, Size, Value, *HeapOrigin.rawId());
  if (PoisonRes != ZE_RESULT_SUCCESS) {
    UR_LOG_L(getContext()->logger, ERR, "Failed to poison {}: {}", Ptr,
             (int)PoisonRes);
  }

  UR_LOG_L(getContext()->logger, INFO,
           "AllocInfo {} (Size={}, Type={}, Origin={})", Ptr, Size,
           ToString(Type), (void *)(uptr)*HeapOrigin.rawId());
  return Res;
}

// Alignment must be at least MSAN_ORIGIN_GRANULARITY; 0 is the driver default.
size_t originAlign(size_t Align) {
  return Align && Align < MSAN_ORIGIN_GRANULARITY ? MSAN_ORIGIN_GRANULARITY
                                                  : Align;
}

ze_memory_type_t memType(const void *Ptr) {
  ze_memory_allocation_properties_t Props{
      ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
  if (Ze.zeMemGetAllocProperties(S.Context, Ptr, &Props, nullptr) !=
      ZE_RESULT_SUCCESS) {
    return ZE_MEMORY_TYPE_UNKNOWN;
  }
  return Props.type;
}

ze_result_t zeMemAllocDeviceHook(ze_context_handle_t hContext,
                                 const ze_device_mem_alloc_desc_t *DDesc,
                                 size_t Size, size_t Align,
                                 ze_device_handle_t hDevice, void **Ptr) {
  Align = originAlign(Align);
  ze_result_t Res =
      ZeReal->zeMemAllocDevice(hContext, DDesc, Size, Align, hDevice, Ptr);
  return afterAlloc(Res, hContext, hDevice, Ptr, Size, HeapType::DeviceUSM);
}

ze_result_t zeMemAllocHostHook(ze_context_handle_t hContext,
                               const ze_host_mem_alloc_desc_t *HDesc,
                               size_t Size, size_t Align, void **Ptr) {
  Align = originAlign(Align);
  ze_result_t Res = ZeReal->zeMemAllocHost(hContext, HDesc, Size, Align, Ptr);
  return afterAlloc(Res, hContext, nullptr, Ptr, Size, HeapType::HostUSM);
}

ze_result_t zeMemAllocSharedHook(ze_context_handle_t hContext,
                                 const ze_device_mem_alloc_desc_t *DDesc,
                                 const ze_host_mem_alloc_desc_t *HDesc,
                                 size_t Size, size_t Align,
                                 ze_device_handle_t hDevice, void **Ptr) {
  Align = originAlign(Align);
  ze_result_t Res = ZeReal->zeMemAllocShared(hContext, DDesc, HDesc, Size,
                                             Align, hDevice, Ptr);
  return afterAlloc(Res, hContext, hDevice, Ptr, Size, HeapType::SharedUSM);
}

// Like msan urEnqueueUSMMemcpy, but blocking.
ze_result_t zeCommandListAppendMemoryCopyHook(
    ze_command_list_handle_t hCmdList, void *Dst, const void *Src, size_t Size,
    ze_event_handle_t hSignal, uint32_t NumWait, ze_event_handle_t *phWait) {
  ze_result_t Res = ZeReal->zeCommandListAppendMemoryCopy(
      hCmdList, Dst, Src, Size, hSignal, NumWait, phWait);
  std::scoped_lock<std::mutex> Guard(S.Mutex);
  if (Res != ZE_RESULT_SUCCESS || !S.Enabled || Size == 0) {
    return Res;
  }
  ZE_CALL(Ze.zeCommandListHostSynchronize(hCmdList, UINT64_MAX));

  ze_memory_type_t SrcType = memType(Src);
  ze_memory_type_t DstType = memType(Dst);
  if (DstType == ZE_MEMORY_TYPE_UNKNOWN) {
    return Res;
  }
  if (SrcType == ZE_MEMORY_TYPE_UNKNOWN) {
    // FIXME: Assume host memory is always initialized memory
    ZE_CALL(poison((uptr)Dst, Size, kMemInitializedMagic));
    return Res;
  }

  uptr DstShadow = S.Shadow.MemToShadow((uptr)Dst);
  ZE_CALL(mapPages(DstShadow, S.Shadow.MemToShadow((uptr)Dst + Size - 1)));
  ZE_CALL(
      copy((void *)DstShadow, (void *)S.Shadow.MemToShadow((uptr)Src), Size));

  if (getContext()->Options.MsanCheckHostAndSharedUSM ||
      (SrcType == ZE_MEMORY_TYPE_DEVICE && DstType == ZE_MEMORY_TYPE_DEVICE)) {
    uptr SrcOriginBegin = S.Shadow.MemToOrigin((uptr)Src);
    uptr SrcOriginEnd =
        S.Shadow.MemToOrigin((uptr)Src + Size - 1) + MSAN_ORIGIN_GRANULARITY;
    uptr DstOrigin = S.Shadow.MemToOrigin((uptr)Dst);
    ZE_CALL(
        mapPages(DstOrigin, DstOrigin + (SrcOriginEnd - SrcOriginBegin) - 1));
    ZE_CALL(copy((void *)DstOrigin, (void *)SrcOriginBegin,
                 SrcOriginEnd - SrcOriginBegin));
  }
  return Res;
}

// Like msan urEnqueueUSMFill, but blocking.
ze_result_t zeCommandListAppendMemoryFillHook(ze_command_list_handle_t hCmdList,
                                              void *Ptr, const void *Pattern,
                                              size_t PatternSize, size_t Size,
                                              ze_event_handle_t hSignal,
                                              uint32_t NumWait,
                                              ze_event_handle_t *phWait) {
  ze_result_t Res = ZeReal->zeCommandListAppendMemoryFill(
      hCmdList, Ptr, Pattern, PatternSize, Size, hSignal, NumWait, phWait);
  std::scoped_lock<std::mutex> Guard(S.Mutex);
  if (Res != ZE_RESULT_SUCCESS || !S.Enabled ||
      memType(Ptr) == ZE_MEMORY_TYPE_UNKNOWN) {
    return Res;
  }
  ZE_CALL(Ze.zeCommandListHostSynchronize(hCmdList, UINT64_MAX));
  // NOTE: No need to set origin, since its shadow is clean
  ZE_CALL(poison((uptr)Ptr, Size, kMemInitializedMagic));
  return Res;
}

// Reads the whole global variable `Name` of Module into Out.
template <typename T>
bool readGlobal(ze_module_handle_t Module, const std::string &Name,
                std::vector<T> &Out) {
  size_t Size = 0;
  void *Ptr = nullptr;
  if (Ze.zeModuleGetGlobalPointer(Module, Name.c_str(), &Size, &Ptr) !=
      ZE_RESULT_SUCCESS) {
    return false;
  }
  assert(Size % sizeof(T) == 0 && "metadata size is not correct");
  Out.resize(Size / sizeof(T));
  return Out.empty() || copy(Out.data(), Ptr, Size) == ZE_RESULT_SUCCESS;
}

// Like MsanInterceptor::registerSpirKernels + registerDeviceGlobals.
ze_result_t zeModuleCreateHook(ze_context_handle_t hContext,
                               ze_device_handle_t hDevice,
                               const ze_module_desc_t *Desc,
                               ze_module_handle_t *phModule,
                               ze_module_build_log_handle_t *phLog) {
  ze_result_t Res =
      ZeReal->zeModuleCreate(hContext, hDevice, Desc, phModule, phLog);
  std::scoped_lock<std::mutex> Guard(S.Mutex);
  if (Res != ZE_RESULT_SUCCESS || !init(hContext, hDevice)) {
    return Res;
  }
  ze_module_handle_t Module = *phModule;
  ModuleRecord &MR = S.Modules[Module];
  MR = {};

  uint32_t Count = 0;
  ZE_CALL(Ze.zeModuleGetKernelNames(Module, &Count, nullptr));
  std::vector<const char *> Names(Count);
  ZE_CALL(Ze.zeModuleGetKernelNames(Module, &Count, Names.data()));

  constexpr std::string_view ModuleIDPrefix = "__sanitizerModule_";
  for (std::string_view Name : Names) {
    if (Name.substr(0, ModuleIDPrefix.size()) != ModuleIDPrefix) {
      continue;
    }
    std::string ID(Name.substr(ModuleIDPrefix.size()));

    std::vector<SpirKernelInfo> SKInfo;
    if (readGlobal(Module, kSPIR_MsanSpirKernelMetadataPrefix + ID, SKInfo)) {
      for (const auto &SKI : SKInfo) {
        if (SKI.Size == 0) {
          continue;
        }
        std::string KernelName(SKI.Size, '\0');
        ZE_CALL(copy(KernelName.data(), (void *)SKI.KernelName, SKI.Size));
        ProgramInfo::KernelMetadata KM{
            bool(SKI.Flags & SanitizedKernelFlags::CHECK_LOCALS),
            bool(SKI.Flags & SanitizedKernelFlags::CHECK_PRIVATES),
            bool(SKI.Flags & SanitizedKernelFlags::MSAN_TRACK_ORIGINS)};
        UR_LOG_L(getContext()->logger, INFO,
                 "SpirKernel(name='{}', isInstrumented={}, "
                 "checkLocals={}, checkPrivates={}, trackOrigins={})",
                 KernelName, true, KM.CheckLocals, KM.CheckPrivates,
                 KM.TrackOrigins);
        MR.Kernels[KernelName] = KM;
      }
    }

    std::vector<DeviceGlobalInfo> GVInfos;
    if (readGlobal(Module, kSPIR_MsanDeviceGlobalMetadataPrefix + ID,
                   GVInfos)) {
      for (const auto &GVInfo : GVInfos) {
        // Only support device global USM
        if (MsanShadowMemoryPVC::IsDeviceUSM(GVInfo.Addr)) {
          ZE_CALL(poison(GVInfo.Addr, GVInfo.Size, kMemInitializedMagic));
          S.CleanShadowSize = std::max(S.CleanShadowSize, GVInfo.Size);
        }
      }
    }
  }
  UR_LOG_L(getContext()->logger, INFO, "Number of sanitized kernel: {}",
           MR.Kernels.size());

  if (!MR.Kernels.empty() &&
      Ze.zeModuleGetGlobalPointer(Module, "__MsanLaunchInfo", nullptr,
                                  &MR.LaunchInfo) != ZE_RESULT_SUCCESS) {
    UR_LOG_L(getContext()->logger, INFO,
             "No __MsanLaunchInfo, maybe empty kernel");
  }
  return Res;
}

ze_result_t zeKernelCreateHook(ze_module_handle_t hModule,
                               const ze_kernel_desc_t *Desc,
                               ze_kernel_handle_t *phKernel) {
  ze_result_t Res = ZeReal->zeKernelCreate(hModule, Desc, phKernel);
  std::scoped_lock<std::mutex> Guard(S.Mutex);
  if (Res != ZE_RESULT_SUCCESS) {
    return Res;
  }
  // Drop a stale record if the handle is reused.
  S.Kernels.erase(*phKernel);
  auto MI = S.Modules.find(hModule);
  if (MI == S.Modules.end()) {
    return Res;
  }
  auto KI = MI->second.Kernels.find(Desc->pKernelName);
  if (KI != MI->second.Kernels.end() && MI->second.LaunchInfo) {
    S.Kernels[*phKernel] = {KI->first, KI->second, MI->second.LaunchInfo};
  }
  return Res;
}

ze_result_t zeKernelSetGroupSizeHook(ze_kernel_handle_t hKernel, uint32_t X,
                                     uint32_t Y, uint32_t Z) {
  ze_result_t Res = ZeReal->zeKernelSetGroupSize(hKernel, X, Y, Z);
  std::scoped_lock<std::mutex> Guard(S.Mutex);
  auto KI = S.Kernels.find(hKernel);
  if (Res == ZE_RESULT_SUCCESS && KI != S.Kernels.end()) {
    KI->second.GroupSize[0] = X;
    KI->second.GroupSize[1] = Y;
    KI->second.GroupSize[2] = Z;
  }
  return Res;
}

// Like MsanInterceptor::prepareLaunch + postLaunchKernel.
ze_result_t zeCommandListAppendLaunchKernelHook(
    ze_command_list_handle_t hCmdList, ze_kernel_handle_t hKernel,
    const ze_group_count_t *GroupCount, ze_event_handle_t hSignal,
    uint32_t NumWait, ze_event_handle_t *phWait) {
  auto Launch = [&] {
    return ZeReal->zeCommandListAppendLaunchKernel(
        hCmdList, hKernel, GroupCount, hSignal, NumWait, phWait);
  };
  std::unique_lock<std::mutex> Guard(S.Mutex);
  auto KI = S.Kernels.find(hKernel);
  uint32_t NumWG = GroupCount->groupCountX * GroupCount->groupCountY *
                   GroupCount->groupCountZ;
  if (KI == S.Kernels.end() || NumWG == 0) {
    Guard.unlock();
    return Launch();
  }
  const KernelRecord &K = KI->second;
  UR_LOG_L(getContext()->logger, INFO,
           "KernelInfo {} (Name=<{}>, CheckLocals={}, CheckPrivates={}, "
           "TrackOrigins={})",
           (void *)hKernel, K.Name, K.Meta.CheckLocals, K.Meta.CheckPrivates,
           K.Meta.TrackOrigins);

  std::vector<void *> ToFree;
  auto Alloc = [&](size_t Size, bool Zero) -> uptr {
    void *Ptr = allocDevice(Size);
    if (Ptr) {
      ToFree.push_back(Ptr);
      const u8 Value = 0;
      if (Zero && fill((uptr)Ptr, &Value, 1, Size) != ZE_RESULT_SUCCESS) {
        return 0;
      }
    }
    return (uptr)Ptr;
  };
  auto Run = [&]() -> ze_result_t {
    MsanRuntimeData Host{};
    Host.GlobalShadowOffset = S.Shadow.ShadowBegin;
    Host.GlobalShadowOffsetEnd = S.Shadow.ShadowEnd;
    Host.DeviceTy = DeviceType::GPU_PVC;
    Host.Debug = getContext()->Options.Debug ? 1 : 0;
    Host.IsRecover = getContext()->Options.Recover ? 1 : 0;

    // Clean shadow: always zero, used for unsupported memory types
    Host.CleanShadow = Alloc(S.CleanShadowSize, true);
    if (!Host.CleanShadow) {
      return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    uint64_t NumWILocal =
        (uint64_t)K.GroupSize[0] * K.GroupSize[1] * K.GroupSize[2];
    ze_kernel_properties_t KP{ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES};
    uint32_t SGSize = S.MinSubGroupSize;
    if (Ze.zeKernelGetProperties(hKernel, &KP) == ZE_RESULT_SUCCESS &&
        KP.requiredSubgroupSize) {
      SGSize = KP.requiredSubgroupSize;
    }
    uint32_t NumSG = ((NumWILocal + SGSize - 1) / SGSize) * NumWG;

    if (K.Meta.CheckLocals) {
      size_t Size = std::min(NumWG, MSAN_MAX_WG_LOCAL) * S.LocalMemorySize;
      if ((Host.LocalShadowOffset = Alloc(Size, true))) {
        Host.LocalShadowOffsetEnd = Host.LocalShadowOffset + Size - 1;
      } else {
        UR_LOG_L(getContext()->logger, WARN,
                 "Skip checking local memory of kernel <{}>", K.Name);
      }
    }

    if (K.Meta.CheckPrivates) {
      NumSG = std::min(NumSG, MSAN_MAX_SG_PRIVATE);
      size_t Size = NumSG * MSAN_PRIVATE_SIZE;
      // No need to clean the private base, it's set by work items on launch
      Host.PrivateBase = (uptr *)Alloc(NumSG * sizeof(uptr), false);
      Host.PrivateShadowOffset = Alloc(Size, true);
      if (Host.PrivateBase && Host.PrivateShadowOffset) {
        Host.PrivateShadowOffsetEnd = Host.PrivateShadowOffset + Size - 1;
      } else {
        UR_LOG_L(getContext()->logger, WARN,
                 "Skip checking private memory of kernel <{}>", K.Name);
        Host.PrivateShadowOffset = 0;
      }
    }

    void *DevicePtr = (void *)Alloc(sizeof(MsanRuntimeData), false);
    if (!DevicePtr) {
      return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    ZE_CALL(copy(DevicePtr, &Host, sizeof(Host)));
    ZE_CALL(copy(K.LaunchInfo, &DevicePtr, sizeof(DevicePtr)));
    UR_LOG_L(getContext()->logger, INFO,
             "LaunchInfo {} (GlobalShadow={}, LocalShadow={}, PrivateBase={}, "
             "PrivateShadow={}, CleanShadow={}, Debug={})",
             DevicePtr, (void *)Host.GlobalShadowOffset,
             (void *)Host.LocalShadowOffset, (void *)Host.PrivateBase,
             (void *)Host.PrivateShadowOffset, (void *)Host.CleanShadow,
             Host.Debug);

    // The runtime data is reached through __MsanLaunchInfo, not an argument
    ZE_CALL(Ze.zeKernelSetIndirectAccess(
        hKernel, ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST |
                     ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE |
                     ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED));

    ZE_CALL(Launch());
    // FIXME: blocking until we can defer the report check to an event
    ZE_CALL(Ze.zeCommandListHostSynchronize(hCmdList, UINT64_MAX));
    ZE_CALL(copy(&Host, DevicePtr, sizeof(Host)));

    if (Host.Report.Flag && ReportUsesUninitializedValue(Host.Report, K.Name)) {
      getMsanInterceptor()->exitWithErrors();
    }
    return ZE_RESULT_SUCCESS;
  };
  ze_result_t Res = Run();
  for (void *Ptr : ToFree) {
    Ze.zeMemFree(S.Context, Ptr);
  }
  return Res;
}

} // namespace

const MsanZeTable *getZeHooks() {
  static const MsanZeTable Hooks = {
#define MSAN_ZE_HOOK(N, P, A) &N##Hook,
      MSAN_ZE_HOOKS(MSAN_ZE_HOOK)
#undef MSAN_ZE_HOOK
  };
  return &Hooks;
}

} // namespace msan
} // namespace ur_sanitizer_layer
