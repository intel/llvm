//===--------- context.cpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <climits>
#include <mutex>
#include <string.h>

#include "context.hpp"
#include "logger/ur_logger.hpp"
#include "queue.hpp"
#include "ur_level_zero.hpp"

namespace ur::level_zero {

ur_result_t urContextCreate(
    /// [in] the number of devices given in phDevices
    uint32_t DeviceCount,
    /// [in][range(0, DeviceCount)] array of handle of devices.
    const ur_device_handle_t *Devices,
    /// [in][optional] pointer to context creation properties.
    const ur_context_properties_t *Properties,
    /// [out] pointer to handle of context object created
    ur_context_handle_t *RetContext) {
  std::ignore = Properties;

  ur_platform_handle_t Platform = Devices[0]->Platform;
  ZeStruct<ze_context_desc_t> ContextDesc{};

  ze_context_handle_t ZeContext{};
  ZE2UR_CALL(zeContextCreate, (Platform->ZeDriver, &ContextDesc, &ZeContext));
  try {
    ur_context_handle_t_ *Context =
        new ur_context_handle_t_(ZeContext, DeviceCount, Devices, true);

    Context->initialize();
    *RetContext = reinterpret_cast<ur_context_handle_t>(Context);
    if (IndirectAccessTrackingEnabled) {
      std::scoped_lock<ur_shared_mutex> Lock(Platform->ContextsMutex);
      Platform->Contexts.push_back(*RetContext);
    }
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (umf_result_t e) {
    return umf::umf2urResult(e);
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urContextRetain(

    /// [in] handle of the context to get a reference of.
    ur_context_handle_t Context) {
  Context->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t urContextRelease(
    /// [in] handle of the context to release.
    ur_context_handle_t Context) {
  ur_platform_handle_t Plt = Context->getPlatform();
  std::unique_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex,
                                                 std::defer_lock);
  if (IndirectAccessTrackingEnabled)
    ContextsLock.lock();

  return ContextReleaseHelper(Context);
}

// Due to a bug with 2D memory copy to and from non-USM pointers, this option is
// disabled by default.
static const bool UseMemcpy2DOperations = [] {
  const char *UrRet = std::getenv("UR_L0_USE_NATIVE_USM_MEMCPY2D");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_USE_NATIVE_USM_MEMCPY2D");
  const char *UseMemcpy2DOperationsFlag =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  if (!UseMemcpy2DOperationsFlag)
    return false;
  return std::atoi(UseMemcpy2DOperationsFlag) > 0;
}();

ur_result_t urContextGetInfo(
    /// [in] handle of the context
    ur_context_handle_t Context,
    /// [in] type of the info to retrieve
    ur_context_info_t ContextInfoType,
    /// [in] the number of bytes of memory pointed to by pContextInfo.
    size_t PropSize,
    /// [out][optional] array of bytes holding the info. if propSize is not
    /// equal to or greater than the real number of bytes needed to return the
    /// info then the ::UR_RESULT_ERROR_INVALID_SIZE error is returned and
    /// pContextInfo is not used.
    void *ContextInfo,
    /// [out][optional] pointer to the actual size in bytes of data queried by
    /// ContextInfoType.
    size_t *PropSizeRet) {
  std::shared_lock<ur_shared_mutex> Lock(Context->Mutex);
  UrReturnHelper ReturnValue(PropSize, ContextInfo, PropSizeRet);
  switch (
      (uint32_t)ContextInfoType) { // cast to avoid warnings on EXT enum values
  case UR_CONTEXT_INFO_DEVICES:
    return ReturnValue(&Context->Devices[0], Context->Devices.size());
  case UR_CONTEXT_INFO_NUM_DEVICES:
    return ReturnValue(uint32_t(Context->Devices.size()));
  case UR_CONTEXT_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{Context->RefCount.load()});
  case UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT:
    // 2D USM memcpy is supported.
    return ReturnValue(uint8_t{UseMemcpy2DOperations});
  case UR_CONTEXT_INFO_USM_FILL2D_SUPPORT:
    // 2D USM fill is not supported.
    return ReturnValue(uint8_t{false});
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES: {

    ur_memory_order_capability_flags_t Capabilities =
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELAXED |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQUIRE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_RELEASE |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_ACQ_REL |
        UR_MEMORY_ORDER_CAPABILITY_FLAG_SEQ_CST;
    return ReturnValue(Capabilities);
  }
  case UR_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES:
  case UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }

  default:
    // TODO: implement other parameters
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
}

ur_result_t urContextGetNativeHandle(
    /// [in] handle of the context.
    ur_context_handle_t Context,
    /// [out] a pointer to the native handle of the context.
    ur_native_handle_t *NativeContext) {
  *NativeContext = reinterpret_cast<ur_native_handle_t>(Context->ZeContext);
  return UR_RESULT_SUCCESS;
}

ur_result_t urContextCreateWithNativeHandle(
    ur_native_handle_t
        /// [in] the native handle of the context.
        NativeContext,
    ur_adapter_handle_t, uint32_t NumDevices, const ur_device_handle_t *Devices,
    const ur_context_native_properties_t *Properties,
    /// [out] pointer to the handle of the context object created.
    ur_context_handle_t *Context) {
  bool OwnNativeHandle = Properties ? Properties->isNativeHandleOwned : false;
  try {
    ze_context_handle_t ZeContext =
        reinterpret_cast<ze_context_handle_t>(NativeContext);
    ur_context_handle_t_ *UrContext = new ur_context_handle_t_(
        ZeContext, NumDevices, Devices, OwnNativeHandle);
    UrContext->initialize();
    *Context = reinterpret_cast<ur_context_handle_t>(UrContext);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urContextSetExtendedDeleter(
    /// [in] handle of the context.
    ur_context_handle_t Context,
    /// [in] Function pointer to extended deleter.
    ur_context_extended_deleter_t Deleter,
    /// [in][out][optional] pointer to data to be passed to callback.
    void *UserData) {
  std::ignore = Context;
  std::ignore = Deleter;
  std::ignore = UserData;
  logger::error(logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
} // namespace ur::level_zero

ur_result_t ur_context_handle_t_::initialize() {

  // Helper lambda to create various USM allocators for a device.
  // Note that the CCS devices and their respective subdevices share a
  // common ze_device_handle and therefore, also share USM allocators.
  auto createUSMAllocators = [this](ur_device_handle_t Device) {
    auto MemProvider = umf::memoryProviderMakeUnique<L0DeviceMemoryProvider>(
                           reinterpret_cast<ur_context_handle_t>(this), Device)
                           .second;
    auto UmfDeviceParamsHandle = getUmfParamsHandle(
        DisjointPoolConfigInstance.Configs[usm::DisjointPoolMemType::Device]);
    DeviceMemPools.emplace(
        std::piecewise_construct, std::make_tuple(Device->ZeDevice),
        std::make_tuple(umf::poolMakeUniqueFromOps(umfDisjointPoolOps(),
                                                   std::move(MemProvider),
                                                   UmfDeviceParamsHandle.get())
                            .second));

    MemProvider = umf::memoryProviderMakeUnique<L0SharedMemoryProvider>(
                      reinterpret_cast<ur_context_handle_t>(this), Device)
                      .second;

    auto UmfSharedParamsHandle = getUmfParamsHandle(
        DisjointPoolConfigInstance.Configs[usm::DisjointPoolMemType::Shared]);
    SharedMemPools.emplace(
        std::piecewise_construct, std::make_tuple(Device->ZeDevice),
        std::make_tuple(umf::poolMakeUniqueFromOps(umfDisjointPoolOps(),
                                                   std::move(MemProvider),
                                                   UmfSharedParamsHandle.get())
                            .second));

    MemProvider = umf::memoryProviderMakeUnique<L0SharedReadOnlyMemoryProvider>(
                      reinterpret_cast<ur_context_handle_t>(this), Device)
                      .second;

    auto UmfSharedROParamsHandle = getUmfParamsHandle(
        DisjointPoolConfigInstance
            .Configs[usm::DisjointPoolMemType::SharedReadOnly]);
    SharedReadOnlyMemPools.emplace(
        std::piecewise_construct, std::make_tuple(Device->ZeDevice),
        std::make_tuple(umf::poolMakeUniqueFromOps(
                            umfDisjointPoolOps(), std::move(MemProvider),
                            UmfSharedROParamsHandle.get())
                            .second));

    MemProvider = umf::memoryProviderMakeUnique<L0DeviceMemoryProvider>(
                      reinterpret_cast<ur_context_handle_t>(this), Device)
                      .second;
    DeviceMemProxyPools.emplace(
        std::piecewise_construct, std::make_tuple(Device->ZeDevice),
        std::make_tuple(
            umf::poolMakeUnique<USMProxyPool>(std::move(MemProvider)).second));

    MemProvider = umf::memoryProviderMakeUnique<L0SharedMemoryProvider>(
                      reinterpret_cast<ur_context_handle_t>(this), Device)
                      .second;
    SharedMemProxyPools.emplace(
        std::piecewise_construct, std::make_tuple(Device->ZeDevice),
        std::make_tuple(
            umf::poolMakeUnique<USMProxyPool>(std::move(MemProvider)).second));

    MemProvider = umf::memoryProviderMakeUnique<L0SharedReadOnlyMemoryProvider>(
                      reinterpret_cast<ur_context_handle_t>(this), Device)
                      .second;
    SharedReadOnlyMemProxyPools.emplace(
        std::piecewise_construct, std::make_tuple(Device->ZeDevice),
        std::make_tuple(
            umf::poolMakeUnique<USMProxyPool>(std::move(MemProvider)).second));
  };

  // Recursive helper to call createUSMAllocators for all sub-devices
  std::function<void(ur_device_handle_t)> createUSMAllocatorsRecursive;
  createUSMAllocatorsRecursive =
      [createUSMAllocators,
       &createUSMAllocatorsRecursive](ur_device_handle_t Device) -> void {
    createUSMAllocators(Device);
    for (auto &SubDevice : Device->SubDevices)
      createUSMAllocatorsRecursive(SubDevice);
  };

  // Create USM pool for each pair (device, context).
  //
  for (auto &Device : Devices) {
    createUSMAllocatorsRecursive(Device);
  }
  // Create USM pool for host. Device and Shared USM allocations
  // are device-specific. Host allocations are not device-dependent therefore
  // we don't need a map with device as key.
  auto MemProvider = umf::memoryProviderMakeUnique<L0HostMemoryProvider>(
                         reinterpret_cast<ur_context_handle_t>(this), nullptr)
                         .second;
  auto UmfHostParamsHandle = getUmfParamsHandle(
      DisjointPoolConfigInstance.Configs[usm::DisjointPoolMemType::Host]);
  HostMemPool =
      umf::poolMakeUniqueFromOps(umfDisjointPoolOps(), std::move(MemProvider),
                                 UmfHostParamsHandle.get())
          .second;

  MemProvider = umf::memoryProviderMakeUnique<L0HostMemoryProvider>(
                    reinterpret_cast<ur_context_handle_t>(this), nullptr)
                    .second;
  HostMemProxyPool =
      umf::poolMakeUnique<USMProxyPool>(std::move(MemProvider)).second;

  // We may allocate memory to this root device so create allocators.
  if (SingleRootDevice &&
      DeviceMemPools.find(SingleRootDevice->ZeDevice) == DeviceMemPools.end()) {
    createUSMAllocators(SingleRootDevice);
  }

  // Create the immediate command list to be used for initializations.
  // Created as synchronous so level-zero performs implicit synchronization and
  // there is no need to query for completion in the plugin
  //
  // TODO: we use Device[0] here as the single immediate command-list
  // for buffer creation and migration. Initialization is in
  // in sync and is always performed to Devices[0] as well but
  // D2D migartion, if no P2P, is broken since it should use
  // immediate command-list for the specfic devices, and this single one.
  //
  ur_device_handle_t Device = SingleRootDevice ? SingleRootDevice : Devices[0];

  // Prefer to use copy engine for initialization copies,
  // if available and allowed (main copy engine with index 0).
  ZeStruct<ze_command_queue_desc_t> ZeCommandQueueDesc;
  const auto &Range = getRangeOfAllowedCopyEngines((ur_device_handle_t)Device);
  ZeCommandQueueDesc.ordinal =
      Device->QueueGroup[ur_device_handle_t_::queue_group_info_t::Compute]
          .ZeOrdinal;
  if (Range.first >= 0 &&
      Device->QueueGroup[ur_device_handle_t_::queue_group_info_t::MainCopy]
              .ZeOrdinal != -1)
    ZeCommandQueueDesc.ordinal =
        Device->QueueGroup[ur_device_handle_t_::queue_group_info_t::MainCopy]
            .ZeOrdinal;

  ZeCommandQueueDesc.index = 0;
  ZeCommandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  if (Device->useDriverInOrderLists() &&
      Device->useDriverCounterBasedEvents()) {
    logger::debug(
        "L0 Synchronous Immediate Command List needed with In Order property.");
    ZeCommandQueueDesc.flags |= ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  }
  ZE2UR_CALL(
      zeCommandListCreateImmediate,
      (ZeContext, Device->ZeDevice, &ZeCommandQueueDesc, &ZeCommandListInit));

  return UR_RESULT_SUCCESS;
}

ur_device_handle_t ur_context_handle_t_::getRootDevice() const {
  assert(Devices.size() > 0);

  if (Devices.size() == 1)
    return Devices[0];

  // Check if we have context with subdevices of the same device (context
  // may include root device itself as well)
  ur_device_handle_t ContextRootDevice =
      Devices[0]->RootDevice ? Devices[0]->RootDevice : Devices[0];

  // For context with sub subdevices, the ContextRootDevice might still
  // not be the root device.
  // Check whether the ContextRootDevice is the subdevice or root device.
  if (ContextRootDevice->isSubDevice()) {
    ContextRootDevice = ContextRootDevice->RootDevice;
  }

  for (auto &Device : Devices) {
    if ((!Device->RootDevice && Device != ContextRootDevice) ||
        (Device->RootDevice && Device->RootDevice != ContextRootDevice)) {
      ContextRootDevice = nullptr;
      break;
    }
  }
  return ContextRootDevice;
}

// Helper function to release the context, a caller must lock the platform-level
// mutex guarding the container with contexts because the context can be removed
// from the list of tracked contexts.
ur_result_t ContextReleaseHelper(ur_context_handle_t Context) {

  if (!Context->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  if (IndirectAccessTrackingEnabled) {
    ur_platform_handle_t Plt = Context->getPlatform();
    auto &Contexts = Plt->Contexts;
    auto It = std::find(Contexts.begin(), Contexts.end(), Context);
    if (It != Contexts.end())
      Contexts.erase(It);
  }
  ze_context_handle_t DestroyZeContext =
      Context->OwnNativeHandle ? Context->ZeContext : nullptr;

  // Clean up any live memory associated with Context
  ur_result_t Result = Context->finalize();

  // We must delete Context first and then destroy zeContext because
  // Context deallocation requires ZeContext in some member deallocation of
  // ur_context_handle_t.
  delete Context;

  // Destruction of some members of ur_context_handle_t uses L0 context
  // and therefore it must be valid at that point.
  // Technically it should be placed to the destructor of ur_context_handle_t
  // but this makes API error handling more complex.
  if (DestroyZeContext) {
    auto ZeResult = ZE_CALL_NOCHECK(zeContextDestroy, (DestroyZeContext));
    // Gracefully handle the case that L0 was already unloaded.
    if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
      return ze2urResult(ZeResult);
  }

  return Result;
}

ur_platform_handle_t ur_context_handle_t_::getPlatform() const {
  return Devices[0]->Platform;
}

ur_result_t ur_context_handle_t_::finalize() {
  // This function is called when ur_context_handle_t is deallocated,
  // urContextRelease. There could be some memory that may have not been
  // deallocated. For example, event and event pool caches would be still alive.

  if (!DisableEventsCaching) {
    std::scoped_lock<ur_mutex> Lock(EventCacheMutex);
    for (auto &EventCache : EventCaches) {
      for (auto &Event : EventCache) {
        auto ZeResult = ZE_CALL_NOCHECK(zeEventDestroy, (Event->ZeEvent));
        Event->ZeEvent = nullptr;
        // Gracefully handle the case that L0 was already unloaded.
        if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
          return ze2urResult(ZeResult);
        delete Event;
      }
      EventCache.clear();
    }
  }
  {
    std::scoped_lock<ur_mutex> Lock(ZeEventPoolCacheMutex);
    for (auto &ZePoolCache : ZeEventPoolCache) {
      for (auto &ZePool : ZePoolCache) {
        auto ZeResult = ZE_CALL_NOCHECK(zeEventPoolDestroy, (ZePool));
        // Gracefully handle the case that L0 was already unloaded.
        if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
          return ze2urResult(ZeResult);
      }
      ZePoolCache.clear();
    }
  }

  // Destroy the command list used for initializations
  auto ZeResult = ZE_CALL_NOCHECK(zeCommandListDestroy, (ZeCommandListInit));
  // Gracefully handle the case that L0 was already unloaded.
  if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
    return ze2urResult(ZeResult);

  std::scoped_lock<ur_mutex> Lock(ZeCommandListCacheMutex);
  for (auto &List : ZeComputeCommandListCache) {
    for (auto &Item : List.second) {
      ze_command_list_handle_t ZeCommandList = Item.first;
      if (ZeCommandList) {
        auto ZeResult = ZE_CALL_NOCHECK(zeCommandListDestroy, (ZeCommandList));
        // Gracefully handle the case that L0 was already unloaded.
        if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
          return ze2urResult(ZeResult);
      }
    }
  }
  for (auto &List : ZeCopyCommandListCache) {
    for (auto &Item : List.second) {
      ze_command_list_handle_t ZeCommandList = Item.first;
      if (ZeCommandList) {
        auto ZeResult = ZE_CALL_NOCHECK(zeCommandListDestroy, (ZeCommandList));
        // Gracefully handle the case that L0 was already unloaded.
        if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
          return ze2urResult(ZeResult);
      }
    }
  }
  return UR_RESULT_SUCCESS;
}

// Maximum number of events that can be present in an event ZePool is captured
// here. Setting it to 256 gave best possible performance for several
// benchmarks.
static const uint32_t MaxNumEventsPerPool = [] {
  const char *UrRet = std::getenv("UR_L0_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL");
  const char *PiRet = std::getenv("ZE_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL");
  const char *MaxNumEventsPerPoolEnv =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  uint32_t Result =
      MaxNumEventsPerPoolEnv ? std::atoi(MaxNumEventsPerPoolEnv) : 256;
  if (Result <= 0)
    Result = 256;
  return Result;
}();

ur_result_t ur_context_handle_t_::getFreeSlotInExistingOrNewPool(
    ze_event_pool_handle_t &Pool, size_t &Index, bool HostVisible,
    bool ProfilingEnabled, ur_device_handle_t Device,
    bool CounterBasedEventEnabled, bool UsingImmCmdList,
    bool InterruptBasedEventEnabled) {
  // Lock while updating event pool machinery.
  std::scoped_lock<ur_mutex> Lock(ZeEventPoolCacheMutex);

  ze_device_handle_t ZeDevice = nullptr;

  if (Device) {
    ZeDevice = Device->ZeDevice;
  }
  std::list<ze_event_pool_handle_t> *ZePoolCache = getZeEventPoolCache(
      HostVisible, ProfilingEnabled, CounterBasedEventEnabled, UsingImmCmdList,
      InterruptBasedEventEnabled, ZeDevice);

  if (!ZePoolCache->empty()) {
    if (NumEventsAvailableInEventPool[ZePoolCache->front()] == 0) {
      if (DisableEventsCaching) {
        // Remove full pool from the cache if events caching is disabled.
        ZePoolCache->erase(ZePoolCache->begin());
      } else {
        // If event caching is enabled then we don't destroy events so there is
        // no need to remove pool from the cache and add it back when it has
        // available slots. Just keep it in the tail of the cache so that all
        // pools can be destroyed during context destruction.
        ZePoolCache->push_front(nullptr);
      }
    }
  }
  if (ZePoolCache->empty()) {
    ZePoolCache->push_back(nullptr);
  }

  // We shall be adding an event to the front pool.
  ze_event_pool_handle_t *ZePool = &ZePoolCache->front();
  Index = 0;
  // Create one event ZePool per MaxNumEventsPerPool events
  if (*ZePool == nullptr) {
    ze_event_pool_counter_based_exp_desc_t counterBasedExt = {
        ZE_STRUCTURE_TYPE_COUNTER_BASED_EVENT_POOL_EXP_DESC, nullptr, 0};

    ze_intel_event_sync_mode_exp_desc_t eventSyncMode = {
        ZE_INTEL_STRUCTURE_TYPE_EVENT_SYNC_MODE_EXP_DESC, nullptr, 0};
    eventSyncMode.syncModeFlags =
        ZE_INTEL_EVENT_SYNC_MODE_EXP_FLAG_LOW_POWER_WAIT |
        ZE_INTEL_EVENT_SYNC_MODE_EXP_FLAG_SIGNAL_INTERRUPT;

    ZeStruct<ze_event_pool_desc_t> ZeEventPoolDesc;
    ZeEventPoolDesc.count = MaxNumEventsPerPool;
    ZeEventPoolDesc.flags = 0;
    ZeEventPoolDesc.pNext = nullptr;
    if (HostVisible)
      ZeEventPoolDesc.flags |= ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    if (ProfilingEnabled)
      ZeEventPoolDesc.flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    logger::debug("ze_event_pool_desc_t flags set to: {}",
                  ZeEventPoolDesc.flags);
    if (CounterBasedEventEnabled) {
      if (UsingImmCmdList) {
        counterBasedExt.flags = ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_IMMEDIATE;
      } else {
        counterBasedExt.flags =
            ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_NON_IMMEDIATE;
      }
      logger::debug("ze_event_pool_desc_t counter based flags set to: {}",
                    counterBasedExt.flags);
      if (InterruptBasedEventEnabled) {
        counterBasedExt.pNext = &eventSyncMode;
      }
      ZeEventPoolDesc.pNext = &counterBasedExt;
    } else if (InterruptBasedEventEnabled) {
      ZeEventPoolDesc.pNext = &eventSyncMode;
    }

    std::vector<ze_device_handle_t> ZeDevices;
    if (ZeDevice) {
      ZeDevices.push_back(ZeDevice);
    } else {
      std::for_each(Devices.begin(), Devices.end(),
                    [&](const ur_device_handle_t &D) {
                      ZeDevices.push_back(D->ZeDevice);
                    });
    }

    ZE2UR_CALL(zeEventPoolCreate, (ZeContext, &ZeEventPoolDesc,
                                   ZeDevices.size(), &ZeDevices[0], ZePool));
    NumEventsAvailableInEventPool[*ZePool] = MaxNumEventsPerPool - 1;
    NumEventsUnreleasedInEventPool[*ZePool] = 1;
  } else {
    Index = MaxNumEventsPerPool - NumEventsAvailableInEventPool[*ZePool];
    --NumEventsAvailableInEventPool[*ZePool];
    ++NumEventsUnreleasedInEventPool[*ZePool];
  }
  Pool = *ZePool;
  return UR_RESULT_SUCCESS;
}

ur_event_handle_t ur_context_handle_t_::getEventFromContextCache(
    bool HostVisible, bool WithProfiling, ur_device_handle_t Device,
    bool CounterBasedEventEnabled, bool InterruptBasedEventEnabled) {
  std::scoped_lock<ur_mutex> Lock(EventCacheMutex);
  auto Cache =
      getEventCache(HostVisible, WithProfiling, Device,
                    CounterBasedEventEnabled, InterruptBasedEventEnabled);
  if (Cache->empty()) {
    logger::info("Cache empty (Host Visible: {}, Profiling: {}, Counter: {}, "
                 "Interrupt: {}, Device: {})",
                 HostVisible, WithProfiling, CounterBasedEventEnabled,
                 InterruptBasedEventEnabled, Device);
    return nullptr;
  }

  auto It = Cache->begin();
  ur_event_handle_t Event = *It;

  Cache->erase(It);
  // We have to reset event before using it.
  Event->reset();

  logger::info("Using {} event (Host Visible: {}, Profiling: {}, Counter: {}, "
               "Interrupt: {}, Device: {}) from cache {}",
               Event, Event->HostVisibleEvent, Event->isProfilingEnabled(),
               Event->CounterBasedEventsEnabled,
               Event->InterruptBasedEventsEnabled, Device, Cache);

  return Event;
}

void ur_context_handle_t_::addEventToContextCache(ur_event_handle_t Event) {
  std::scoped_lock<ur_mutex> Lock(EventCacheMutex);
  ur_device_handle_t Device = nullptr;

  if (!Event->IsMultiDevice && Event->UrQueue) {
    Device = Event->UrQueue->Device;
  }

  auto Cache = getEventCache(
      Event->isHostVisible(), Event->isProfilingEnabled(), Device,
      Event->CounterBasedEventsEnabled, Event->InterruptBasedEventsEnabled);
  logger::info("Inserting {} event (Host Visible: {}, Profiling: {}, Counter: "
               "{}, Device: {}) into cache {}",
               Event, Event->HostVisibleEvent, Event->isProfilingEnabled(),
               Event->CounterBasedEventsEnabled, Device, Cache);
  Cache->emplace_back(Event);
}

ur_result_t
ur_context_handle_t_::decrementUnreleasedEventsInPool(ur_event_handle_t Event) {
  std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex, std::defer_lock);
  std::scoped_lock<ur_mutex, std::shared_lock<ur_shared_mutex>> LockAll(
      ZeEventPoolCacheMutex, EventLock);
  if (!Event->ZeEventPool) {
    // This must be an interop event created on a users's pool.
    // Do nothing.
    return UR_RESULT_SUCCESS;
  }

  ze_device_handle_t ZeDevice = nullptr;
  bool UsingImmediateCommandlists =
      !Event->UrQueue || Event->UrQueue->UsingImmCmdLists;

  if (!Event->IsMultiDevice && Event->UrQueue) {
    ZeDevice = Event->UrQueue->Device->ZeDevice;
  }

  std::list<ze_event_pool_handle_t> *ZePoolCache = getZeEventPoolCache(
      Event->isHostVisible(), Event->isProfilingEnabled(),
      Event->CounterBasedEventsEnabled, UsingImmediateCommandlists,
      Event->InterruptBasedEventsEnabled, ZeDevice);

  // Put the empty pool to the cache of the pools.
  if (NumEventsUnreleasedInEventPool[Event->ZeEventPool] == 0)
    die("Invalid event release: event pool doesn't have unreleased events");
  if (--NumEventsUnreleasedInEventPool[Event->ZeEventPool] == 0) {
    if (ZePoolCache->front() != Event->ZeEventPool) {
      ZePoolCache->push_back(Event->ZeEventPool);
    }
    NumEventsAvailableInEventPool[Event->ZeEventPool] = MaxNumEventsPerPool;
  }

  return UR_RESULT_SUCCESS;
}

// Get value of the threshold for number of active command lists allowed before
// we start heuristically cleaning them up.
static const size_t CmdListsCleanupThreshold = [] {
  const char *UrRet = std::getenv("UR_L0_COMMANDLISTS_CLEANUP_THRESHOLD");
  const char *PiRet =
      std::getenv("SYCL_PI_LEVEL_ZERO_COMMANDLISTS_CLEANUP_THRESHOLD");
  const char *CmdListsCleanupThresholdStr =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  static constexpr int Default = 20;
  if (!CmdListsCleanupThresholdStr)
    return Default;

  int Threshold = std::atoi(CmdListsCleanupThresholdStr);

  // Basically disable threshold if negative value is provided.
  if (Threshold < 0)
    return INT_MAX;

  return Threshold;
}();

// Retrieve an available command list to be used in a PI call.
ur_result_t ur_context_handle_t_::getAvailableCommandList(
    ur_queue_handle_t Queue, ur_command_list_ptr_t &CommandList,
    bool UseCopyEngine, uint32_t NumEventsInWaitList,
    const ur_event_handle_t *EventWaitList, bool AllowBatching,
    ze_command_queue_handle_t *ForcedCmdQueue) {
  // Immediate commandlists have been pre-allocated and are always available.
  if (Queue->UsingImmCmdLists) {
    CommandList = Queue->getQueueGroup(UseCopyEngine).getImmCmdList();
    if (CommandList->second.EventList.size() >=
        Queue->getImmdCmmdListsEventCleanupThreshold()) {
      std::vector<ur_event_handle_t> EventListToCleanup;
      Queue->resetCommandList(CommandList, false, EventListToCleanup);
      CleanupEventListFromResetCmdList(EventListToCleanup, true);
    }
    UR_CALL(Queue->insertStartBarrierIfDiscardEventsMode(CommandList));
    if (auto Res = Queue->insertActiveBarriers(CommandList, UseCopyEngine))
      return Res;
    return UR_RESULT_SUCCESS;
  } else {
    // Cleanup regular command-lists if there are too many.
    // It handles the case that the queue is not synced to the host
    // for a long time and we want to reclaim the command-lists for
    // use by other queues.
    if (Queue->CommandListMap.size() > CmdListsCleanupThreshold) {
      resetCommandLists(Queue);
    }
  }

  auto &CommandBatch =
      UseCopyEngine ? Queue->CopyCommandBatch : Queue->ComputeCommandBatch;
  // Handle batching of commands
  // First see if there is an command-list open for batching commands
  // for this queue.
  if (Queue->hasOpenCommandList(UseCopyEngine)) {
    if (AllowBatching) {
      bool batchingAllowed = true;
      if (ForcedCmdQueue &&
          CommandBatch.OpenCommandList->second.ZeQueue != *ForcedCmdQueue) {
        // Current open batch doesn't match the forced command queue
        batchingAllowed = false;
      }
      if (!UrL0OutOfOrderIntegratedSignalEvent &&
          Queue->Device->isIntegrated()) {
        batchingAllowed = eventCanBeBatched(Queue, UseCopyEngine,
                                            NumEventsInWaitList, EventWaitList);
      }
      if (batchingAllowed) {
        CommandList = CommandBatch.OpenCommandList;
        UR_CALL(Queue->insertStartBarrierIfDiscardEventsMode(CommandList));
        return UR_RESULT_SUCCESS;
      }
    }
    // If this command isn't allowed to be batched or doesn't match the forced
    // command queue, then we need to go ahead and execute what is already in
    // the batched list, and then go on to process this. On exit from
    // executeOpenCommandList OpenCommandList will be invalidated.
    if (auto Res = Queue->executeOpenCommandList(UseCopyEngine))
      return Res;
    // Note that active barriers do not need to be inserted here as they will
    // have been enqueued into the command-list when they were created.
  }

  // Create/Reuse the command list, because in Level Zero commands are added to
  // the command lists, and later are then added to the command queue.
  // Each command list is paired with an associated fence to track when the
  // command list is available for reuse.
  ur_result_t ur_result = UR_RESULT_ERROR_OUT_OF_RESOURCES;

  // Initally, we need to check if a command list has already been created
  // on this device that is available for use. If so, then reuse that
  // Level-Zero Command List and Fence for this PI call.
  {
    // Make sure to acquire the lock before checking the size, or there
    // will be a race condition.
    std::scoped_lock<ur_mutex> Lock(Queue->Context->ZeCommandListCacheMutex);
    // Under mutex since operator[] does insertion on the first usage for
    // every unique ZeDevice.
    auto &ZeCommandListCache =
        UseCopyEngine
            ? Queue->Context->ZeCopyCommandListCache[Queue->Device->ZeDevice]
            : Queue->Context
                  ->ZeComputeCommandListCache[Queue->Device->ZeDevice];

    for (auto ZeCommandListIt = ZeCommandListCache.begin();
         ZeCommandListIt != ZeCommandListCache.end(); ++ZeCommandListIt) {
      // If this is an InOrder Queue, then only allow lists which are in order.
      if (Queue->Device->useDriverInOrderLists() && Queue->isInOrderQueue() &&
          !(ZeCommandListIt->second.InOrderList)) {
        continue;
      }
      // Only allow to reuse Regular Command Lists
      if (ZeCommandListIt->second.IsImmediate) {
        continue;
      }
      auto &ZeCommandList = ZeCommandListIt->first;
      auto it = Queue->CommandListMap.find(ZeCommandList);
      if (it != Queue->CommandListMap.end()) {
        if (ForcedCmdQueue && *ForcedCmdQueue != it->second.ZeQueue)
          continue;
        CommandList = it;
        if (CommandList->second.ZeFence != nullptr)
          CommandList->second.ZeFenceInUse = true;
      } else {
        // If there is a command list available on this context, but it
        // wasn't yet used in this queue then create a new entry in this
        // queue's map to hold the fence and other associated command
        // list information.
        auto &QGroup = Queue->getQueueGroup(UseCopyEngine);
        uint32_t QueueGroupOrdinal;
        auto &ZeCommandQueue = ForcedCmdQueue
                                   ? *ForcedCmdQueue
                                   : QGroup.getZeQueue(&QueueGroupOrdinal);
        if (ForcedCmdQueue)
          QueueGroupOrdinal = QGroup.getCmdQueueOrdinal(ZeCommandQueue);

        ze_fence_handle_t ZeFence;
        ZeStruct<ze_fence_desc_t> ZeFenceDesc;
        ZE2UR_CALL(zeFenceCreate, (ZeCommandQueue, &ZeFenceDesc, &ZeFence));
        ZeStruct<ze_command_queue_desc_t> ZeQueueDesc;
        ZeQueueDesc.ordinal = QueueGroupOrdinal;

        CommandList =
            Queue->CommandListMap
                .emplace(ZeCommandList,
                         ur_command_list_info_t(
                             ZeFence, true, false, ZeCommandQueue, ZeQueueDesc,
                             Queue->useCompletionBatching(), true /*CanReuse */,
                             ZeCommandListIt->second.InOrderList,
                             ZeCommandListIt->second.IsImmediate))
                .first;
      }
      ZeCommandListCache.erase(ZeCommandListIt);
      if (auto Res = Queue->insertStartBarrierIfDiscardEventsMode(CommandList))
        return Res;
      if (auto Res = Queue->insertActiveBarriers(CommandList, UseCopyEngine))
        return Res;
      return UR_RESULT_SUCCESS;
    }
  }

  // If there are no available command lists in the cache, then we check for
  // command lists that have already signalled, but have not been added to the
  // available list yet. Each command list has a fence associated which tracks
  // if a command list has completed dispatch of its commands and is ready for
  // reuse. If a command list is found to have been signalled, then the
  // command list & fence are reset and we return.
  for (auto it = Queue->CommandListMap.begin();
       it != Queue->CommandListMap.end(); ++it) {
    // Make sure this is the command list type needed.
    if (UseCopyEngine != it->second.isCopy(Queue))
      continue;

    // If this is an InOrder Queue, then only allow lists which are in order.
    if (Queue->Device->useDriverInOrderLists() && Queue->isInOrderQueue() &&
        !(it->second.IsInOrderList)) {
      continue;
    }

    ze_result_t ZeResult =
        ZE_CALL_NOCHECK(zeFenceQueryStatus, (it->second.ZeFence));
    if (ZeResult == ZE_RESULT_SUCCESS) {
      std::vector<ur_event_handle_t> EventListToCleanup;
      Queue->resetCommandList(it, false, EventListToCleanup);
      CleanupEventListFromResetCmdList(EventListToCleanup,
                                       true /* QueueLocked */);
      CommandList = it;
      CommandList->second.ZeFenceInUse = true;
      if (auto Res = Queue->insertStartBarrierIfDiscardEventsMode(CommandList))
        return Res;
      return UR_RESULT_SUCCESS;
    }
  }

  // If there are no available command lists nor signalled command lists,
  // then we must create another command list.
  ur_result = Queue->createCommandList(UseCopyEngine, CommandList);
  CommandList->second.ZeFenceInUse = true;
  return ur_result;
}

bool ur_context_handle_t_::isValidDevice(ur_device_handle_t Device) const {
  while (Device) {
    if (std::find(Devices.begin(), Devices.end(), Device) != Devices.end())
      return true;
    Device = Device->RootDevice;
  }
  return false;
}

const std::vector<ur_device_handle_t> &
ur_context_handle_t_::getDevices() const {
  return Devices;
}

ze_context_handle_t ur_context_handle_t_::getZeHandle() const {
  return ZeContext;
}
