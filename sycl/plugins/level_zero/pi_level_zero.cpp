//===-------- pi_level_zero.cpp - Level Zero Plugin --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

/// \file pi_level_zero.cpp
/// Implementation of Level Zero Plugin.
///
/// \ingroup sycl_pi_level_zero

#include "pi_level_zero.hpp"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <sycl/detail/pi.h>
#include <sycl/detail/spinlock.hpp>
#include <utility>

#include <zet_api.h>

#include "ur/usm_allocator_config.hpp"
#include "ur_bindings.hpp"

extern "C" {
// Forward declarartions.
static pi_result piQueueReleaseInternal(pi_queue Queue);
static pi_result piEventReleaseInternal(pi_event Event);
static pi_result EventCreate(pi_context Context, pi_queue Queue,
                             bool HostVisible, pi_event *RetEvent);
}

// Defined in tracing.cpp
void enableZeTracing();
void disableZeTracing();

namespace {

// This is an experimental option to test performance of device to device copy
// operations on copy engines (versus compute engine)
static const bool UseCopyEngineForD2DCopy = [] {
  const char *UrRet = std::getenv("UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY");
  const char *PiRet =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY");
  const char *CopyEngineForD2DCopy = UrRet ? UrRet : (PiRet ? PiRet : nullptr);

  return (CopyEngineForD2DCopy && (std::stoi(CopyEngineForD2DCopy) != 0));
}();

// This is an experimental option that allows the use of copy engine, if
// available in the device, in Level Zero plugin for copy operations submitted
// to an in-order queue. The default is 1.
static const bool UseCopyEngineForInOrderQueue = [] {
  const char *UrRet = std::getenv("UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE");
  const char *PiRet =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE");
  const char *CopyEngineForInOrderQueue =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);

  return (!CopyEngineForInOrderQueue ||
          (std::stoi(CopyEngineForInOrderQueue) != 0));
}();

// This is an experimental option that allows the use of multiple command lists
// when submitting barriers. The default is 0.
static const bool UseMultipleCmdlistBarriers = [] {
  const char *UrRet = std::getenv("UR_L0_USE_MULTIPLE_COMMANDLIST_BARRIERS");
  const char *PiRet =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_MULTIPLE_COMMANDLIST_BARRIERS");
  const char *UseMultipleCmdlistBarriersFlag =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);

  if (!UseMultipleCmdlistBarriersFlag)
    return true;
  return std::stoi(UseMultipleCmdlistBarriersFlag) > 0;
}();

// This is an experimental option that allows to disable caching of events in
// the context.
static const bool DisableEventsCaching = [] {
  const char *UrRet = std::getenv("UR_L0_DISABLE_EVENTS_CACHING");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_DISABLE_EVENTS_CACHING");
  const char *DisableEventsCachingFlag =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);

  if (!DisableEventsCachingFlag)
    return false;
  return std::stoi(DisableEventsCachingFlag) != 0;
}();

// This is an experimental option that allows reset and reuse of uncompleted
// events in the in-order queue with discard_events property.
static const bool ReuseDiscardedEvents = [] {
  const char *UrRet = std::getenv("UR_L0_REUSE_DISCARDED_EVENTS");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_REUSE_DISCARDED_EVENTS");
  const char *ReuseDiscardedEventsFlag =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);

  if (!ReuseDiscardedEventsFlag)
    return true;
  return std::stoi(ReuseDiscardedEventsFlag) > 0;
}();

// Due to a bug with 2D memory copy to and from non-USM pointers, this option is
// disabled by default.
static const bool UseMemcpy2DOperations = [] {
  const char *UrRet = std::getenv("UR_L0_USE_NATIVE_USM_MEMCPY2D");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_USE_NATIVE_USM_MEMCPY2D");
  const char *UseMemcpy2DOperationsFlag =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);

  if (!UseMemcpy2DOperationsFlag)
    return false;
  return std::stoi(UseMemcpy2DOperationsFlag) > 0;
}();

// Map from L0 to PI result.
static inline pi_result mapError(ze_result_t Result) {
  return ur2piResult(ze2urResult(Result));
}

// Trace a call to Level-Zero RT
#define ZE_CALL(ZeName, ZeArgs)                                                \
  {                                                                            \
    ze_result_t ZeResult = ZeName ZeArgs;                                      \
    if (auto Result = ZeCall().doCall(ZeResult, #ZeName, #ZeArgs, true))       \
      return mapError(Result);                                                 \
  }

// Trace an internal PI call; returns in case of an error.
#define PI_CALL(Call)                                                          \
  {                                                                            \
    if (PrintTrace)                                                            \
      fprintf(stderr, "PI ---> %s\n", #Call);                                  \
    pi_result Result = (Call);                                                 \
    if (Result != PI_SUCCESS)                                                  \
      return Result;                                                           \
  }

// Controls if we should choose doing eager initialization
// to make it happen on warmup paths and have the reportable
// paths be less likely affected.
//
static bool doEagerInit = [] {
  const char *UrRet = std::getenv("UR_L0_EAGER_INIT");
  const char *PiRet = std::getenv("SYCL_EAGER_INIT");
  const char *EagerInit = UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  return EagerInit ? std::atoi(EagerInit) != 0 : false;
}();

// Maximum number of events that can be present in an event ZePool is captured
// here. Setting it to 256 gave best possible performance for several
// benchmarks.
static const pi_uint32 MaxNumEventsPerPool = [] {
  const char *UrRet = std::getenv("UR_L0_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL");
  const char *PiRet = std::getenv("ZE_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL");
  const char *MaxNumEventsPerPoolEnv =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);

  pi_uint32 Result =
      MaxNumEventsPerPoolEnv ? std::atoi(MaxNumEventsPerPoolEnv) : 256;
  if (Result <= 0)
    Result = 256;
  return Result;
}();

// Helper function to implement zeHostSynchronize.
// The behavior is to avoid infinite wait during host sync under ZE_DEBUG.
// This allows for a much more responsive debugging of hangs.
//
template <typename T, typename Func>
ze_result_t zeHostSynchronizeImpl(Func Api, T Handle) {
  if (!UrL0Debug) {
    return Api(Handle, UINT64_MAX);
  }

  ze_result_t R;
  while ((R = Api(Handle, 1000)) == ZE_RESULT_NOT_READY)
    ;
  return R;
}

// Template function to do various types of host synchronizations.
// This is intended to be used instead of direct calls to specific
// Level-Zero synchronization APIs.
//
template <typename T> ze_result_t zeHostSynchronize(T Handle);
template <> ze_result_t zeHostSynchronize(ze_event_handle_t Handle) {
  return zeHostSynchronizeImpl(zeEventHostSynchronize, Handle);
}
template <> ze_result_t zeHostSynchronize(ze_command_queue_handle_t Handle) {
  return zeHostSynchronizeImpl(zeCommandQueueSynchronize, Handle);
}

} // anonymous namespace

// UR_L0_LEVEL_ZERO_USE_COMPUTE_ENGINE can be set to an integer (>=0) in
// which case all compute commands will be submitted to the command-queue
// with the given index in the compute command group. If it is instead set
// to negative then all available compute engines may be used.
//
// The default value is "0".
//
static const std::pair<int, int> getRangeOfAllowedComputeEngines() {
  const char *UrRet = std::getenv("UR_L0_USE_COMPUTE_ENGINE");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_USE_COMPUTE_ENGINE");
  const char *EnvVar = UrRet ? UrRet : (PiRet ? PiRet : nullptr);

  // If the environment variable is not set only use "0" CCS for now.
  // TODO: allow all CCSs when HW support is complete.
  if (!EnvVar)
    return std::pair<int, int>(0, 0);

  auto EnvVarValue = std::atoi(EnvVar);
  if (EnvVarValue >= 0) {
    return std::pair<int, int>(EnvVarValue, EnvVarValue);
  }

  return std::pair<int, int>(0, INT_MAX);
}

pi_platform _pi_context::getPlatform() const { return Devices[0]->Platform; }

bool _pi_context::isValidDevice(pi_device Device) const {
  while (Device) {
    if (std::find(Devices.begin(), Devices.end(), Device) != Devices.end())
      return true;
    Device = Device->RootDevice;
  }
  return false;
}

pi_result
_pi_context::getFreeSlotInExistingOrNewPool(ze_event_pool_handle_t &Pool,
                                            size_t &Index, bool HostVisible,
                                            bool ProfilingEnabled) {
  // Lock while updating event pool machinery.
  std::scoped_lock<ur_mutex> Lock(ZeEventPoolCacheMutex);

  std::list<ze_event_pool_handle_t> *ZePoolCache =
      getZeEventPoolCache(HostVisible, ProfilingEnabled);

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
    ZeStruct<ze_event_pool_desc_t> ZeEventPoolDesc;
    ZeEventPoolDesc.count = MaxNumEventsPerPool;
    ZeEventPoolDesc.flags = 0;
    if (HostVisible)
      ZeEventPoolDesc.flags |= ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    if (ProfilingEnabled)
      ZeEventPoolDesc.flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    urPrint("ze_event_pool_desc_t flags set to: %d\n", ZeEventPoolDesc.flags);

    std::vector<ze_device_handle_t> ZeDevices;
    std::for_each(Devices.begin(), Devices.end(), [&](const pi_device &D) {
      ZeDevices.push_back(D->ZeDevice);
    });

    ZE_CALL(zeEventPoolCreate, (ZeContext, &ZeEventPoolDesc, ZeDevices.size(),
                                &ZeDevices[0], ZePool));
    NumEventsAvailableInEventPool[*ZePool] = MaxNumEventsPerPool - 1;
    NumEventsUnreleasedInEventPool[*ZePool] = 1;
  } else {
    Index = MaxNumEventsPerPool - NumEventsAvailableInEventPool[*ZePool];
    --NumEventsAvailableInEventPool[*ZePool];
    ++NumEventsUnreleasedInEventPool[*ZePool];
  }
  Pool = *ZePool;
  return PI_SUCCESS;
}

pi_result _pi_context::decrementUnreleasedEventsInPool(pi_event Event) {
  std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex, std::defer_lock);
  std::scoped_lock<ur_mutex, std::shared_lock<ur_shared_mutex>> LockAll(
      ZeEventPoolCacheMutex, EventLock);
  if (!Event->ZeEventPool) {
    // This must be an interop event created on a users's pool.
    // Do nothing.
    return PI_SUCCESS;
  }

  std::list<ze_event_pool_handle_t> *ZePoolCache =
      getZeEventPoolCache(Event->isHostVisible(), Event->isProfilingEnabled());

  // Put the empty pool to the cache of the pools.
  if (NumEventsUnreleasedInEventPool[Event->ZeEventPool] == 0)
    die("Invalid event release: event pool doesn't have unreleased events");
  if (--NumEventsUnreleasedInEventPool[Event->ZeEventPool] == 0) {
    if (ZePoolCache->front() != Event->ZeEventPool) {
      ZePoolCache->push_back(Event->ZeEventPool);
    }
    NumEventsAvailableInEventPool[Event->ZeEventPool] = MaxNumEventsPerPool;
  }

  return PI_SUCCESS;
}

// Forward declarations
static pi_result enqueueMemCopyHelper(pi_command_type CommandType,
                                      pi_queue Queue, void *Dst,
                                      pi_bool BlockingWrite, size_t Size,
                                      const void *Src,
                                      pi_uint32 NumEventsInWaitList,
                                      const pi_event *EventWaitList,
                                      pi_event *Event, bool PreferCopyEngine);

static pi_result enqueueMemCopyRectHelper(
    pi_command_type CommandType, pi_queue Queue, const void *SrcBuffer,
    void *DstBuffer, pi_buff_rect_offset SrcOrigin,
    pi_buff_rect_offset DstOrigin, pi_buff_rect_region Region,
    size_t SrcRowPitch, size_t DstRowPitch, size_t SrcSlicePitch,
    size_t DstSlicePitch, pi_bool Blocking, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event,
    bool PreferCopyEngine = false);

bool _pi_queue::doReuseDiscardedEvents() {
  return ReuseDiscardedEvents && isInOrderQueue() && isDiscardEvents();
}

pi_result _pi_queue::resetDiscardedEvent(pi_command_list_ptr_t CommandList) {
  if (LastCommandEvent && LastCommandEvent->IsDiscarded) {
    ZE_CALL(zeCommandListAppendBarrier,
            (CommandList->first, nullptr, 1, &(LastCommandEvent->ZeEvent)));
    ZE_CALL(zeCommandListAppendEventReset,
            (CommandList->first, LastCommandEvent->ZeEvent));

    // Create new pi_event but with the same ze_event_handle_t. We are going
    // to use this pi_event for the next command with discarded event.
    pi_event PiEvent;
    try {
      PiEvent = new _pi_event(LastCommandEvent->ZeEvent,
                              LastCommandEvent->ZeEventPool, Context,
                              PI_COMMAND_TYPE_USER, true);
    } catch (const std::bad_alloc &) {
      return PI_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }

    if (LastCommandEvent->isHostVisible())
      PiEvent->HostVisibleEvent = PiEvent;

    PI_CALL(addEventToQueueCache(PiEvent));
  }

  return PI_SUCCESS;
}

// This helper function creates a pi_event and associate a pi_queue.
// Note that the caller of this function must have acquired lock on the Queue
// that is passed in.
// \param Queue pi_queue to associate with a new event.
// \param Event a pointer to hold the newly created pi_event
// \param CommandType various command type determined by the caller
// \param CommandList is the command list where the event is added
// \param IsInternal tells if the event is internal, i.e. visible in the L0
//        plugin only.
// \param HostVisible tells if the event must be created in the
//        host-visible pool. If not set then this function will decide.
inline static pi_result
createEventAndAssociateQueue(pi_queue Queue, pi_event *Event,
                             pi_command_type CommandType,
                             pi_command_list_ptr_t CommandList, bool IsInternal,
                             std::optional<bool> HostVisible = std::nullopt) {

  if (!HostVisible.has_value()) {
    // Internal/discarded events do not need host-scope visibility.
    HostVisible =
        IsInternal ? false : Queue->Device->ZeEventsScope == AllHostVisible;
  }

  // If event is discarded then try to get event from the queue cache.
  *Event =
      IsInternal ? Queue->getEventFromQueueCache(HostVisible.value()) : nullptr;

  if (*Event == nullptr)
    PI_CALL(EventCreate(Queue->Context, Queue, HostVisible.value(), Event));

  (*Event)->Queue = Queue;
  (*Event)->CommandType = CommandType;
  (*Event)->IsDiscarded = IsInternal;
  (*Event)->CommandList = CommandList;
  // Discarded event doesn't own ze_event, it is used by multiple pi_event
  // objects. We destroy corresponding ze_event by releasing events from the
  // events cache at queue destruction. Event in the cache owns the Level Zero
  // event.
  if (IsInternal)
    (*Event)->OwnZeEvent = false;

  // Append this Event to the CommandList, if any
  if (CommandList != Queue->CommandListMap.end()) {
    CommandList->second.append(*Event);
    (*Event)->RefCount.increment();
  }

  // We need to increment the reference counter here to avoid pi_queue
  // being released before the associated pi_event is released because
  // piEventRelease requires access to the associated pi_queue.
  // In piEventRelease, the reference counter of the Queue is decremented
  // to release it.
  Queue->RefCount.increment();

  // SYCL RT does not track completion of the events, so it could
  // release a PI event as soon as that's not being waited in the app.
  // But we have to ensure that the event is not destroyed before
  // it is really signalled, so retain it explicitly here and
  // release in CleanupCompletedEvent(Event).
  // If the event is internal then don't increment the reference count as this
  // event will not be waited/released by SYCL RT, so it must be destroyed by
  // EventRelease in resetCommandList.
  if (!IsInternal)
    PI_CALL(piEventRetain(*Event));

  return PI_SUCCESS;
}

pi_result _pi_queue::signalEventFromCmdListIfLastEventDiscarded(
    pi_command_list_ptr_t CommandList) {
  // We signal new event at the end of command list only if we have queue with
  // discard_events property and the last command event is discarded.
  if (!(doReuseDiscardedEvents() && LastCommandEvent &&
        LastCommandEvent->IsDiscarded))
    return PI_SUCCESS;

  // NOTE: We create this "glue" event not as internal so it is not
  // participating in the discarded events reset/reuse logic, but
  // with no host-visibility since it is not going to be waited
  // from the host.
  pi_event Event;
  PI_CALL(createEventAndAssociateQueue(
      this, &Event, PI_COMMAND_TYPE_USER, CommandList,
      /* IsInternal */ false, /* HostVisible */ false));
  PI_CALL(piEventReleaseInternal(Event));
  LastCommandEvent = Event;

  ZE_CALL(zeCommandListAppendSignalEvent, (CommandList->first, Event->ZeEvent));
  return PI_SUCCESS;
}

pi_event _pi_queue::getEventFromQueueCache(bool HostVisible) {
  auto Cache = HostVisible ? &EventCaches[0] : &EventCaches[1];

  // If we don't have any events, return nullptr.
  // If we have only a single event then it was used by the last command and we
  // can't use it now because we have to enforce round robin between two events.
  if (Cache->size() < 2)
    return nullptr;

  // If there are two events then return an event from the beginning of the list
  // since event of the last command is added to the end of the list.
  auto It = Cache->begin();
  pi_event RetEvent = *It;
  Cache->erase(It);
  return RetEvent;
}

pi_result _pi_queue::addEventToQueueCache(pi_event Event) {
  auto Cache = Event->isHostVisible() ? &EventCaches[0] : &EventCaches[1];
  Cache->emplace_back(Event);
  return PI_SUCCESS;
}

// Get value of the threshold for number of events in immediate command lists.
// If number of events in the immediate command list exceeds this threshold then
// cleanup process for those events is executed.
static const size_t ImmCmdListsEventCleanupThreshold = [] {
  const char *UrRet =
      std::getenv("UR_L0_IMMEDIATE_COMMANDLISTS_EVENT_CLEANUP_THRESHOLD");
  const char *PiRet = std::getenv(
      "SYCL_PI_LEVEL_ZERO_IMMEDIATE_COMMANDLISTS_EVENT_CLEANUP_THRESHOLD");
  const char *ImmCmdListsEventCleanupThresholdStr =
      UrRet ? UrRet : (PiRet ? PiRet : nullptr);

  static constexpr int Default = 1000;
  if (!ImmCmdListsEventCleanupThresholdStr)
    return Default;

  int Threshold = std::atoi(ImmCmdListsEventCleanupThresholdStr);

  // Basically disable threshold if negative value is provided.
  if (Threshold < 0)
    return INT_MAX;

  return Threshold;
}();

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

pi_device _pi_context::getRootDevice() const {
  assert(Devices.size() > 0);

  if (Devices.size() == 1)
    return Devices[0];

  // Check if we have context with subdevices of the same device (context
  // may include root device itself as well)
  pi_device ContextRootDevice =
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

pi_result _pi_context::initialize() {

  // Helper lambda to create various USM allocators for a device.
  // Note that the CCS devices and their respective subdevices share a
  // common ze_device_handle and therefore, also share USM allocators.
  auto createUSMAllocators = [this](pi_device Device) {
    SharedMemAllocContexts.emplace(
        std::piecewise_construct, std::make_tuple(Device->ZeDevice),
        std::make_tuple(
            std::unique_ptr<SystemMemory>(
                new USMSharedMemoryAlloc(this, Device)),
            USMAllocatorConfigInstance.Configs[usm_settings::MemType::Shared]));

    SharedReadOnlyMemAllocContexts.emplace(
        std::piecewise_construct, std::make_tuple(Device->ZeDevice),
        std::make_tuple(std::unique_ptr<SystemMemory>(
                            new USMSharedReadOnlyMemoryAlloc(this, Device)),
                        USMAllocatorConfigInstance
                            .Configs[usm_settings::MemType::SharedReadOnly]));

    DeviceMemAllocContexts.emplace(
        std::piecewise_construct, std::make_tuple(Device->ZeDevice),
        std::make_tuple(
            std::unique_ptr<SystemMemory>(
                new USMDeviceMemoryAlloc(this, Device)),
            USMAllocatorConfigInstance.Configs[usm_settings::MemType::Device]));
  };

  // Recursive helper to call createUSMAllocators for all sub-devices
  std::function<void(pi_device)> createUSMAllocatorsRecursive;
  createUSMAllocatorsRecursive =
      [createUSMAllocators,
       &createUSMAllocatorsRecursive](pi_device Device) -> void {
    createUSMAllocators(Device);
    for (auto &SubDevice : Device->SubDevices)
      createUSMAllocatorsRecursive(SubDevice);
  };

  // Create USM allocator context for each pair (device, context).
  //
  for (auto &Device : Devices) {
    createUSMAllocatorsRecursive(Device);
  }
  // Create USM allocator context for host. Device and Shared USM allocations
  // are device-specific. Host allocations are not device-dependent therefore
  // we don't need a map with device as key.
  HostMemAllocContext = std::make_unique<USMAllocContext>(
      std::unique_ptr<SystemMemory>(new USMHostMemoryAlloc(this)),
      USMAllocatorConfigInstance.Configs[usm_settings::MemType::Host]);

  // We may allocate memory to this root device so create allocators.
  if (SingleRootDevice &&
      DeviceMemAllocContexts.find(SingleRootDevice->ZeDevice) ==
          DeviceMemAllocContexts.end()) {
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
  pi_device Device = SingleRootDevice ? SingleRootDevice : Devices[0];

  // Prefer to use copy engine for initialization copies,
  // if available and allowed (main copy engine with index 0).
  ZeStruct<ze_command_queue_desc_t> ZeCommandQueueDesc;
  const auto &Range = getRangeOfAllowedCopyEngines((ur_device_handle_t)Device);
  ZeCommandQueueDesc.ordinal =
      Device->QueueGroup[_pi_device::queue_group_info_t::Compute].ZeOrdinal;
  if (Range.first >= 0 &&
      Device->QueueGroup[_pi_device::queue_group_info_t::MainCopy].ZeOrdinal !=
          -1)
    ZeCommandQueueDesc.ordinal =
        Device->QueueGroup[_pi_device::queue_group_info_t::MainCopy].ZeOrdinal;

  ZeCommandQueueDesc.index = 0;
  ZeCommandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  ZE_CALL(
      zeCommandListCreateImmediate,
      (ZeContext, Device->ZeDevice, &ZeCommandQueueDesc, &ZeCommandListInit));
  return PI_SUCCESS;
}

pi_result _pi_context::finalize() {
  // This function is called when pi_context is deallocated, piContextRelease.
  // There could be some memory that may have not been deallocated.
  // For example, event and event pool caches would be still alive.

  if (!DisableEventsCaching) {
    std::scoped_lock<ur_mutex> Lock(EventCacheMutex);
    for (auto &EventCache : EventCaches) {
      for (auto &Event : EventCache) {
        auto ZeResult = ZE_CALL_NOCHECK(zeEventDestroy, (Event->ZeEvent));
        // Gracefully handle the case that L0 was already unloaded.
        if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
          return mapError(ZeResult);

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
          return mapError(ZeResult);
      }
      ZePoolCache.clear();
    }
  }

  // Destroy the command list used for initializations
  auto ZeResult = ZE_CALL_NOCHECK(zeCommandListDestroy, (ZeCommandListInit));
  // Gracefully handle the case that L0 was already unloaded.
  if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
    return mapError(ZeResult);

  std::scoped_lock<ur_mutex> Lock(ZeCommandListCacheMutex);
  for (auto &List : ZeComputeCommandListCache) {
    for (ze_command_list_handle_t &ZeCommandList : List.second) {
      if (ZeCommandList) {
        auto ZeResult = ZE_CALL_NOCHECK(zeCommandListDestroy, (ZeCommandList));
        // Gracefully handle the case that L0 was already unloaded.
        if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
          return mapError(ZeResult);
      }
    }
  }
  for (auto &List : ZeCopyCommandListCache) {
    for (ze_command_list_handle_t &ZeCommandList : List.second) {
      if (ZeCommandList) {
        auto ZeResult = ZE_CALL_NOCHECK(zeCommandListDestroy, (ZeCommandList));
        // Gracefully handle the case that L0 was already unloaded.
        if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
          return mapError(ZeResult);
      }
    }
  }
  return PI_SUCCESS;
}

bool pi_command_list_info_t::isCopy(pi_queue Queue) const {
  return ZeQueueGroupOrdinal !=
         (uint32_t)Queue->Device
             ->QueueGroup[_pi_device::queue_group_info_t::type::Compute]
             .ZeOrdinal;
}

bool _pi_queue::isInOrderQueue() const {
  // If out-of-order queue property is not set, then this is a in-order queue.
  return ((this->Properties & PI_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) ==
          0);
}

bool _pi_queue::isDiscardEvents() const {
  return ((this->Properties & PI_EXT_ONEAPI_QUEUE_FLAG_DISCARD_EVENTS) != 0);
}

bool _pi_queue::isPriorityLow() const {
  return ((this->Properties & PI_EXT_ONEAPI_QUEUE_FLAG_PRIORITY_LOW) != 0);
}

bool _pi_queue::isPriorityHigh() const {
  return ((this->Properties & PI_EXT_ONEAPI_QUEUE_FLAG_PRIORITY_HIGH) != 0);
}

pi_result _pi_queue::resetCommandList(pi_command_list_ptr_t CommandList,
                                      bool MakeAvailable,
                                      std::vector<pi_event> &EventListToCleanup,
                                      bool CheckStatus) {
  bool UseCopyEngine = CommandList->second.isCopy(this);

  // Immediate commandlists do not have an associated fence.
  if (CommandList->second.ZeFence != nullptr) {
    // Fence had been signalled meaning the associated command-list completed.
    // Reset the fence and put the command list into a cache for reuse in PI
    // calls.
    ZE_CALL(zeFenceReset, (CommandList->second.ZeFence));
    ZE_CALL(zeCommandListReset, (CommandList->first));
    CommandList->second.ZeFenceInUse = false;
    CommandList->second.IsClosed = false;
  }

  auto &EventList = CommandList->second.EventList;
  // Check if standard commandlist or fully synced in-order queue.
  // If one of those conditions is met then we are sure that all events are
  // completed so we don't need to check event status.
  if (!CheckStatus || CommandList->second.ZeFence != nullptr ||
      (isInOrderQueue() && !LastCommandEvent)) {
    // Remember all the events in this command list which needs to be
    // released/cleaned up and clear event list associated with command list.
    std::move(std::begin(EventList), std::end(EventList),
              std::back_inserter(EventListToCleanup));
    EventList.clear();
  } else if (!isDiscardEvents()) {
    // If events in the queue are discarded then we can't check their status.
    // Helper for checking of event completion
    auto EventCompleted = [](pi_event Event) -> bool {
      std::scoped_lock<ur_shared_mutex> EventLock(Event->Mutex);
      ze_result_t ZeResult =
          Event->Completed
              ? ZE_RESULT_SUCCESS
              : ZE_CALL_NOCHECK(zeEventQueryStatus, (Event->ZeEvent));
      return ZeResult == ZE_RESULT_SUCCESS;
    };
    // Handle in-order specially as we can just in few checks (with binary
    // search) a completed event and then all events before it are also
    // done.
    if (isInOrderQueue()) {
      size_t Bisect = EventList.size();
      size_t Iter = 0;
      for (auto it = EventList.rbegin(); it != EventList.rend(); ++Iter) {
        if (!EventCompleted(*it)) {
          if (Bisect > 1 && Iter < 3) { // Heuristically limit by 3 checks
            Bisect >>= 1;
            it += Bisect;
            continue;
          }
          break;
        }
        // Bulk move of event up to "it" to the list ready for cleanup
        std::move(it, EventList.rend(), std::back_inserter(EventListToCleanup));
        EventList.erase(EventList.begin(), it.base());
        break;
      }
      return PI_SUCCESS;
    }
    // For immediate commandlist reset only those events that have signalled.
    for (auto it = EventList.begin(); it != EventList.end();) {
      // Break early as soon as we found first incomplete event because next
      // events are submitted even later. We are not trying to find all
      // completed events here because it may be costly. I.e. we are checking
      // only elements which are most likely completed because they were
      // submitted earlier. It is guaranteed that all events will be eventually
      // cleaned up at queue sync/release.
      if (!EventCompleted(*it))
        break;

      EventListToCleanup.push_back(std::move((*it)));
      it = EventList.erase(it);
    }
  }

  // Standard commandlists move in and out of the cache as they are recycled.
  // Immediate commandlists are always available.
  if (CommandList->second.ZeFence != nullptr && MakeAvailable) {
    std::scoped_lock<ur_mutex> Lock(this->Context->ZeCommandListCacheMutex);
    auto &ZeCommandListCache =
        UseCopyEngine
            ? this->Context->ZeCopyCommandListCache[this->Device->ZeDevice]
            : this->Context->ZeComputeCommandListCache[this->Device->ZeDevice];
    ZeCommandListCache.push_back(CommandList->first);
  }

  return PI_SUCCESS;
}

// Configuration of the command-list batching.
struct zeCommandListBatchConfig {
  // Default value of 0. This specifies to use dynamic batch size adjustment.
  // Other values will try to collect specified amount of commands.
  pi_uint32 Size{0};

  // If doing dynamic batching, specifies start batch size.
  pi_uint32 DynamicSizeStart{4};

  // The maximum size for dynamic batch.
  pi_uint32 DynamicSizeMax{64};

  // The step size for dynamic batch increases.
  pi_uint32 DynamicSizeStep{1};

  // Thresholds for when increase batch size (number of closed early is small
  // and number of closed full is high).
  pi_uint32 NumTimesClosedEarlyThreshold{3};
  pi_uint32 NumTimesClosedFullThreshold{8};

  // Tells the starting size of a batch.
  pi_uint32 startSize() const { return Size > 0 ? Size : DynamicSizeStart; }
  // Tells is we are doing dynamic batch size adjustment.
  bool dynamic() const { return Size == 0; }
};

// Helper function to initialize static variables that holds batch config info
// for compute and copy command batching.
static const zeCommandListBatchConfig ZeCommandListBatchConfig(bool IsCopy) {
  zeCommandListBatchConfig Config{}; // default initialize

  // Default value of 0. This specifies to use dynamic batch size adjustment.
  const char *UrRet = nullptr;
  const char *PiRet = nullptr;
  if (IsCopy) {
    UrRet = std::getenv("UR_L0_COPY_BATCH_SIZE");
    PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_COPY_BATCH_SIZE");
  } else {
    UrRet = std::getenv("UR_L0_BATCH_SIZE");
    PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_BATCH_SIZE");
  }
  const char *BatchSizeStr = UrRet ? UrRet : (PiRet ? PiRet : nullptr);

  if (BatchSizeStr) {
    pi_int32 BatchSizeStrVal = std::atoi(BatchSizeStr);
    // Level Zero may only support a limted number of commands per command
    // list.  The actual upper limit is not specified by the Level Zero
    // Specification.  For now we allow an arbitrary upper limit.
    if (BatchSizeStrVal > 0) {
      Config.Size = BatchSizeStrVal;
    } else if (BatchSizeStrVal == 0) {
      Config.Size = 0;
      // We are requested to do dynamic batching. Collect specifics, if any.
      // The extended format supported is ":" separated values.
      //
      // NOTE: these extra settings are experimental and are intended to
      // be used only for finding a better default heuristic.
      //
      std::string BatchConfig(BatchSizeStr);
      size_t Ord = 0;
      size_t Pos = 0;
      while (true) {
        if (++Ord > 5)
          break;

        Pos = BatchConfig.find(":", Pos);
        if (Pos == std::string::npos)
          break;
        ++Pos; // past the ":"

        pi_uint32 Val;
        try {
          Val = std::stoi(BatchConfig.substr(Pos));
        } catch (...) {
          if (IsCopy)
            urPrint("UR_L0_COPY_BATCH_SIZE: failed to parse value\n");
          else
            urPrint("UR_L0_BATCH_SIZE: failed to parse value\n");
          break;
        }
        switch (Ord) {
        case 1:
          Config.DynamicSizeStart = Val;
          break;
        case 2:
          Config.DynamicSizeMax = Val;
          break;
        case 3:
          Config.DynamicSizeStep = Val;
          break;
        case 4:
          Config.NumTimesClosedEarlyThreshold = Val;
          break;
        case 5:
          Config.NumTimesClosedFullThreshold = Val;
          break;
        default:
          die("Unexpected batch config");
        }
        if (IsCopy)
          urPrint("UR_L0_COPY_BATCH_SIZE: dynamic batch param "
                  "#%d: %d\n",
                  (int)Ord, (int)Val);
        else
          urPrint("UR_L0_BATCH_SIZE: dynamic batch param #%d: %d\n", (int)Ord,
                  (int)Val);
      };

    } else {
      // Negative batch sizes are silently ignored.
      if (IsCopy)
        urPrint("UR_L0_COPY_BATCH_SIZE: ignored negative value\n");
      else
        urPrint("UR_L0_BATCH_SIZE: ignored negative value\n");
    }
  }
  return Config;
}

// Static variable that holds batch config info for compute command batching.
static const zeCommandListBatchConfig ZeCommandListBatchComputeConfig = [] {
  using IsCopy = bool;
  return ZeCommandListBatchConfig(IsCopy{false});
}();

// Static variable that holds batch config info for copy command batching.
static const zeCommandListBatchConfig ZeCommandListBatchCopyConfig = [] {
  using IsCopy = bool;
  return ZeCommandListBatchConfig(IsCopy{true});
}();

_pi_queue::_pi_queue(std::vector<ze_command_queue_handle_t> &ComputeQueues,
                     std::vector<ze_command_queue_handle_t> &CopyQueues,
                     pi_context Context, pi_device Device,
                     bool OwnZeCommandQueue,
                     pi_queue_properties PiQueueProperties,
                     int ForceComputeIndex)
    : Context{Context}, Device{Device}, OwnZeCommandQueue{OwnZeCommandQueue},
      Properties(PiQueueProperties) {
  UsingImmCmdLists = Device->useImmediateCommandLists();
  urPrint("ImmCmdList setting (%s)\n", (UsingImmCmdLists ? "YES" : "NO"));

  // Compute group initialization.
  // First, see if the queue's device allows for round-robin or it is
  // fixed to one particular compute CCS (it is so for sub-sub-devices).
  auto &ComputeQueueGroupInfo = Device->QueueGroup[queue_type::Compute];
  pi_queue_group_t ComputeQueueGroup{this, queue_type::Compute};
  ComputeQueueGroup.ZeQueues = ComputeQueues;
  // Create space to hold immediate commandlists corresponding to the
  // ZeQueues
  if (UsingImmCmdLists) {
    ComputeQueueGroup.ImmCmdLists = std::vector<pi_command_list_ptr_t>(
        ComputeQueueGroup.ZeQueues.size(), CommandListMap.end());
  }
  if (ComputeQueueGroupInfo.ZeIndex >= 0) {
    // Sub-sub-device

    // sycl::ext::intel::property::queue::compute_index works with any
    // backend/device by allowing single zero index if multiple compute CCSes
    // are not supported. Sub-sub-device falls into the same bucket.
    assert(ForceComputeIndex <= 0);
    ComputeQueueGroup.LowerIndex = ComputeQueueGroupInfo.ZeIndex;
    ComputeQueueGroup.UpperIndex = ComputeQueueGroupInfo.ZeIndex;
    ComputeQueueGroup.NextIndex = ComputeQueueGroupInfo.ZeIndex;
  } else if (ForceComputeIndex >= 0) {
    ComputeQueueGroup.LowerIndex = ForceComputeIndex;
    ComputeQueueGroup.UpperIndex = ForceComputeIndex;
    ComputeQueueGroup.NextIndex = ForceComputeIndex;
  } else {
    // Set-up to round-robin across allowed range of engines.
    uint32_t FilterLowerIndex = getRangeOfAllowedComputeEngines().first;
    uint32_t FilterUpperIndex = getRangeOfAllowedComputeEngines().second;
    FilterUpperIndex = std::min((size_t)FilterUpperIndex,
                                FilterLowerIndex + ComputeQueues.size() - 1);
    if (FilterLowerIndex <= FilterUpperIndex) {
      ComputeQueueGroup.LowerIndex = FilterLowerIndex;
      ComputeQueueGroup.UpperIndex = FilterUpperIndex;
      ComputeQueueGroup.NextIndex = ComputeQueueGroup.LowerIndex;
    } else {
      die("No compute queue available/allowed.");
    }
  }
  if (UsingImmCmdLists) {
    // Create space to hold immediate commandlists corresponding to the
    // ZeQueues
    ComputeQueueGroup.ImmCmdLists = std::vector<pi_command_list_ptr_t>(
        ComputeQueueGroup.ZeQueues.size(), CommandListMap.end());
  }
  ComputeQueueGroupsByTID.set(ComputeQueueGroup);

  // Copy group initialization.
  pi_queue_group_t CopyQueueGroup{this, queue_type::MainCopy};
  const auto &Range = getRangeOfAllowedCopyEngines((ur_device_handle_t)Device);
  if (Range.first < 0 || Range.second < 0) {
    // We are asked not to use copy engines, just do nothing.
    // Leave CopyQueueGroup.ZeQueues empty, and it won't be used.
  } else {
    uint32_t FilterLowerIndex = Range.first;
    uint32_t FilterUpperIndex = Range.second;
    FilterUpperIndex = std::min((size_t)FilterUpperIndex,
                                FilterLowerIndex + CopyQueues.size() - 1);
    if (FilterLowerIndex <= FilterUpperIndex) {
      CopyQueueGroup.ZeQueues = CopyQueues;
      CopyQueueGroup.LowerIndex = FilterLowerIndex;
      CopyQueueGroup.UpperIndex = FilterUpperIndex;
      CopyQueueGroup.NextIndex = CopyQueueGroup.LowerIndex;
      // Create space to hold immediate commandlists corresponding to the
      // ZeQueues
      if (UsingImmCmdLists) {
        CopyQueueGroup.ImmCmdLists = std::vector<pi_command_list_ptr_t>(
            CopyQueueGroup.ZeQueues.size(), CommandListMap.end());
      }
    }
  }
  CopyQueueGroupsByTID.set(CopyQueueGroup);

  // Initialize compute/copy command batches.
  ComputeCommandBatch.OpenCommandList = CommandListMap.end();
  CopyCommandBatch.OpenCommandList = CommandListMap.end();
  ComputeCommandBatch.QueueBatchSize =
      ZeCommandListBatchComputeConfig.startSize();
  CopyCommandBatch.QueueBatchSize = ZeCommandListBatchCopyConfig.startSize();
}

static pi_result CleanupCompletedEvent(pi_event Event,
                                       bool QueueLocked = false);

// Helper function to perform the necessary cleanup of the events from reset cmd
// list.
static pi_result
CleanupEventListFromResetCmdList(std::vector<pi_event> &EventListToCleanup,
                                 bool QueueLocked = false) {
  for (auto &Event : EventListToCleanup) {
    // We don't need to synchronize the events since the fence associated with
    // the command list was synchronized.
    {
      std::scoped_lock<ur_shared_mutex> EventLock(Event->Mutex);
      Event->Completed = true;
    }
    PI_CALL(CleanupCompletedEvent(Event, QueueLocked));
    // This event was removed from the command list, so decrement ref count
    // (it was incremented when they were added to the command list).
    PI_CALL(piEventReleaseInternal(Event));
  }
  return PI_SUCCESS;
}

/// @brief Cleanup events in the immediate lists of the queue.
/// @param Queue Queue where events need to be cleaned up.
/// @param QueueLocked Indicates if the queue mutex is locked by caller.
/// @param QueueSynced 'true' if queue was synchronized before the
/// call and no other commands were submitted after synchronization, 'false'
/// otherwise.
/// @param CompletedEvent Hint providing an event which was synchronized before
/// the call, in case of in-order queue it allows to cleanup all preceding
/// events.
/// @return PI_SUCCESS if successful, PI error code otherwise.
static pi_result CleanupEventsInImmCmdLists(pi_queue Queue,
                                            bool QueueLocked = false,
                                            bool QueueSynced = false,
                                            pi_event CompletedEvent = nullptr) {
  // Handle only immediate command lists here.
  if (!Queue || !Queue->UsingImmCmdLists)
    return PI_SUCCESS;

  std::vector<pi_event> EventListToCleanup;
  {
    std::unique_lock<ur_shared_mutex> QueueLock(Queue->Mutex, std::defer_lock);
    if (!QueueLocked)
      QueueLock.lock();
    // If queue is locked and fully synchronized then cleanup all events.
    // If queue is not locked then by this time there may be new submitted
    // commands so we can't do full cleanup.
    if (QueueLocked &&
        (QueueSynced || (Queue->isInOrderQueue() &&
                         (CompletedEvent == Queue->LastCommandEvent ||
                          !Queue->LastCommandEvent)))) {
      Queue->LastCommandEvent = nullptr;
      for (auto &&It = Queue->CommandListMap.begin();
           It != Queue->CommandListMap.end(); ++It) {
        PI_CALL(Queue->resetCommandList(It, true, EventListToCleanup,
                                        /* CheckStatus */ false));
      }
    } else if (Queue->isInOrderQueue() && CompletedEvent) {
      // If the queue is in-order and we have information about completed event
      // then cleanup all events in the command list preceding to CompletedEvent
      // including itself.

      // Check that the comleted event has associated command list.
      if (!(CompletedEvent->CommandList &&
            CompletedEvent->CommandList.value() != Queue->CommandListMap.end()))
        return PI_SUCCESS;

      auto &CmdListEvents =
          CompletedEvent->CommandList.value()->second.EventList;
      auto CompletedEventIt =
          std::find(CmdListEvents.begin(), CmdListEvents.end(), CompletedEvent);
      if (CompletedEventIt != CmdListEvents.end()) {
        // We can cleanup all events prior to the completed event in this
        // command list and completed event itself.
        // TODO: we can potentially cleanup more events here by finding
        // completed events on another command lists, but it is currently not
        // implemented.
        std::move(std::begin(CmdListEvents), CompletedEventIt + 1,
                  std::back_inserter(EventListToCleanup));
        CmdListEvents.erase(CmdListEvents.begin(), CompletedEventIt + 1);
      }
    } else {
      // Fallback to resetCommandList over all command lists.
      for (auto &&It = Queue->CommandListMap.begin();
           It != Queue->CommandListMap.end(); ++It) {
        PI_CALL(Queue->resetCommandList(It, true, EventListToCleanup,
                                        /* CheckStatus */ true));
      }
    }
  }
  PI_CALL(CleanupEventListFromResetCmdList(EventListToCleanup, QueueLocked));
  return PI_SUCCESS;
}

/// @brief Reset signalled command lists in the queue and put them to the cache
/// of command lists. Also cleanup events associated with signalled command
/// lists. Queue must be locked by the caller for modification.
/// @param Queue Queue where we look for signalled command lists and cleanup
/// events.
/// @return PI_SUCCESS if successful, PI error code otherwise.
static pi_result resetCommandLists(pi_queue Queue) {
  // Handle immediate command lists here, they don't need to be reset and we
  // only need to cleanup events.
  if (Queue->UsingImmCmdLists) {
    PI_CALL(CleanupEventsInImmCmdLists(Queue, true /*locked*/));
    return PI_SUCCESS;
  }

  // We need events to be cleaned up out of scope where queue is locked to avoid
  // nested locks, because event cleanup requires event to be locked. Nested
  // locks are hard to control and can cause deadlocks if mutexes are locked in
  // different order.
  std::vector<pi_event> EventListToCleanup;

  // We check for command lists that have been already signalled, but have not
  // been added to the available list yet. Each command list has a fence
  // associated which tracks if a command list has completed dispatch of its
  // commands and is ready for reuse. If a command list is found to have been
  // signalled, then the command list & fence are reset and command list is
  // returned to the command list cache. All events associated with command
  // list are cleaned up if command list was reset.
  for (auto &&it = Queue->CommandListMap.begin();
       it != Queue->CommandListMap.end(); ++it) {
    // Immediate commandlists don't use a fence and are handled separately
    // above.
    assert(it->second.ZeFence != nullptr);
    // It is possible that the fence was already noted as signalled and
    // reset. In that case the ZeFenceInUse flag will be false.
    if (it->second.ZeFenceInUse) {
      ze_result_t ZeResult =
          ZE_CALL_NOCHECK(zeFenceQueryStatus, (it->second.ZeFence));
      if (ZeResult == ZE_RESULT_SUCCESS)
        PI_CALL(Queue->resetCommandList(it, true, EventListToCleanup));
    }
  }
  CleanupEventListFromResetCmdList(EventListToCleanup, true /*locked*/);
  return PI_SUCCESS;
}

// Retrieve an available command list to be used in a PI call.
pi_result _pi_context::getAvailableCommandList(
    pi_queue Queue, pi_command_list_ptr_t &CommandList, bool UseCopyEngine,
    bool AllowBatching, ze_command_queue_handle_t *ForcedCmdQueue) {
  // Immediate commandlists have been pre-allocated and are always available.
  if (Queue->UsingImmCmdLists) {
    CommandList = Queue->getQueueGroup(UseCopyEngine).getImmCmdList();
    if (CommandList->second.EventList.size() >
        ImmCmdListsEventCleanupThreshold) {
      std::vector<pi_event> EventListToCleanup;
      Queue->resetCommandList(CommandList, false, EventListToCleanup);
      CleanupEventListFromResetCmdList(EventListToCleanup, true);
    }
    PI_CALL(Queue->insertStartBarrierIfDiscardEventsMode(CommandList));
    if (auto Res = Queue->insertActiveBarriers(CommandList, UseCopyEngine))
      return Res;
    return PI_SUCCESS;
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
      CommandList = CommandBatch.OpenCommandList;
      PI_CALL(Queue->insertStartBarrierIfDiscardEventsMode(CommandList));
      return PI_SUCCESS;
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
  _pi_result pi_result = PI_ERROR_OUT_OF_RESOURCES;

  // Initally, we need to check if a command list has already been created
  // on this device that is available for use. If so, then reuse that
  // Level-Zero Command List and Fence for this PI call.
  {
    // Make sure to acquire the lock before checking the size, or there
    // will be a race condition.
    std::scoped_lock<ur_mutex> Lock(Queue->Context->ZeCommandListCacheMutex);
    // Under mutex since operator[] does insertion on the first usage for every
    // unique ZeDevice.
    auto &ZeCommandListCache =
        UseCopyEngine
            ? Queue->Context->ZeCopyCommandListCache[Queue->Device->ZeDevice]
            : Queue->Context
                  ->ZeComputeCommandListCache[Queue->Device->ZeDevice];

    for (auto ZeCommandListIt = ZeCommandListCache.begin();
         ZeCommandListIt != ZeCommandListCache.end(); ++ZeCommandListIt) {
      auto &ZeCommandList = *ZeCommandListIt;
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
        ZE_CALL(zeFenceCreate, (ZeCommandQueue, &ZeFenceDesc, &ZeFence));
        CommandList = Queue->CommandListMap
                          .emplace(ZeCommandList,
                                   pi_command_list_info_t{ZeFence, true, false,
                                                          ZeCommandQueue,
                                                          QueueGroupOrdinal})
                          .first;
      }
      ZeCommandListCache.erase(ZeCommandListIt);
      if (auto Res = Queue->insertStartBarrierIfDiscardEventsMode(CommandList))
        return Res;
      if (auto Res = Queue->insertActiveBarriers(CommandList, UseCopyEngine))
        return Res;
      return PI_SUCCESS;
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

    ze_result_t ZeResult =
        ZE_CALL_NOCHECK(zeFenceQueryStatus, (it->second.ZeFence));
    if (ZeResult == ZE_RESULT_SUCCESS) {
      std::vector<pi_event> EventListToCleanup;
      Queue->resetCommandList(it, false, EventListToCleanup);
      CleanupEventListFromResetCmdList(EventListToCleanup,
                                       true /* QueueLocked */);
      CommandList = it;
      CommandList->second.ZeFenceInUse = true;
      if (auto Res = Queue->insertStartBarrierIfDiscardEventsMode(CommandList))
        return Res;
      return PI_SUCCESS;
    }
  }

  // If there are no available command lists nor signalled command lists,
  // then we must create another command list.
  pi_result = Queue->createCommandList(UseCopyEngine, CommandList);
  CommandList->second.ZeFenceInUse = true;
  return pi_result;
}

_pi_queue::pi_queue_group_t &_pi_queue::getQueueGroup(bool UseCopyEngine) {
  auto &Map = (UseCopyEngine ? CopyQueueGroupsByTID : ComputeQueueGroupsByTID);
  return Map.get();
}

// Helper function to create a new command-list to this queue and associated
// fence tracking its completion. This command list & fence are added to the
// map of command lists in this queue with ZeFenceInUse = false.
// The caller must hold a lock of the queue already.
pi_result
_pi_queue::createCommandList(bool UseCopyEngine,
                             pi_command_list_ptr_t &CommandList,
                             ze_command_queue_handle_t *ForcedCmdQueue) {

  ze_fence_handle_t ZeFence;
  ZeStruct<ze_fence_desc_t> ZeFenceDesc;
  ze_command_list_handle_t ZeCommandList;

  uint32_t QueueGroupOrdinal;
  auto &QGroup = getQueueGroup(UseCopyEngine);
  auto &ZeCommandQueue =
      ForcedCmdQueue ? *ForcedCmdQueue : QGroup.getZeQueue(&QueueGroupOrdinal);
  if (ForcedCmdQueue)
    QueueGroupOrdinal = QGroup.getCmdQueueOrdinal(ZeCommandQueue);

  ZeStruct<ze_command_list_desc_t> ZeCommandListDesc;
  ZeCommandListDesc.commandQueueGroupOrdinal = QueueGroupOrdinal;

  ZE_CALL(zeCommandListCreate, (Context->ZeContext, Device->ZeDevice,
                                &ZeCommandListDesc, &ZeCommandList));

  ZE_CALL(zeFenceCreate, (ZeCommandQueue, &ZeFenceDesc, &ZeFence));
  std::tie(CommandList, std::ignore) = CommandListMap.insert(
      std::pair<ze_command_list_handle_t, pi_command_list_info_t>(
          ZeCommandList,
          {ZeFence, false, false, ZeCommandQueue, QueueGroupOrdinal}));

  PI_CALL(insertStartBarrierIfDiscardEventsMode(CommandList));
  PI_CALL(insertActiveBarriers(CommandList, UseCopyEngine));
  return PI_SUCCESS;
}

void _pi_queue::adjustBatchSizeForFullBatch(bool IsCopy) {
  auto &CommandBatch = IsCopy ? CopyCommandBatch : ComputeCommandBatch;
  auto &ZeCommandListBatchConfig =
      IsCopy ? ZeCommandListBatchCopyConfig : ZeCommandListBatchComputeConfig;
  pi_uint32 &QueueBatchSize = CommandBatch.QueueBatchSize;
  // QueueBatchSize of 0 means never allow batching.
  if (QueueBatchSize == 0 || !ZeCommandListBatchConfig.dynamic())
    return;
  CommandBatch.NumTimesClosedFull += 1;

  // If the number of times the list has been closed early is low, and
  // the number of times it has been closed full is high, then raise
  // the batching size slowly. Don't raise it if it is already pretty
  // high.
  if (CommandBatch.NumTimesClosedEarly <=
          ZeCommandListBatchConfig.NumTimesClosedEarlyThreshold &&
      CommandBatch.NumTimesClosedFull >
          ZeCommandListBatchConfig.NumTimesClosedFullThreshold) {
    if (QueueBatchSize < ZeCommandListBatchConfig.DynamicSizeMax) {
      QueueBatchSize += ZeCommandListBatchConfig.DynamicSizeStep;
      urPrint("Raising QueueBatchSize to %d\n", QueueBatchSize);
    }
    CommandBatch.NumTimesClosedEarly = 0;
    CommandBatch.NumTimesClosedFull = 0;
  }
}

void _pi_queue::adjustBatchSizeForPartialBatch(bool IsCopy) {
  auto &CommandBatch = IsCopy ? CopyCommandBatch : ComputeCommandBatch;
  auto &ZeCommandListBatchConfig =
      IsCopy ? ZeCommandListBatchCopyConfig : ZeCommandListBatchComputeConfig;
  pi_uint32 &QueueBatchSize = CommandBatch.QueueBatchSize;
  // QueueBatchSize of 0 means never allow batching.
  if (QueueBatchSize == 0 || !ZeCommandListBatchConfig.dynamic())
    return;
  CommandBatch.NumTimesClosedEarly += 1;

  // If we are closing early more than about 3x the number of times
  // it is closing full, lower the batch size to the value of the
  // current open command list. This is trying to quickly get to a
  // batch size that will be able to be closed full at least once
  // in a while.
  if (CommandBatch.NumTimesClosedEarly >
      (CommandBatch.NumTimesClosedFull + 1) * 3) {
    QueueBatchSize = CommandBatch.OpenCommandList->second.size() - 1;
    if (QueueBatchSize < 1)
      QueueBatchSize = 1;
    urPrint("Lowering QueueBatchSize to %d\n", QueueBatchSize);
    CommandBatch.NumTimesClosedEarly = 0;
    CommandBatch.NumTimesClosedFull = 0;
  }
}

void _pi_queue::CaptureIndirectAccesses() {
  for (auto &Kernel : KernelsToBeSubmitted) {
    if (!Kernel->hasIndirectAccess())
      continue;

    auto &Contexts = Device->Platform->Contexts;
    for (auto &Ctx : Contexts) {
      for (auto &Elem : Ctx->MemAllocs) {
        const auto &Pair = Kernel->MemAllocs.insert(&Elem);
        // Kernel is referencing this memory allocation from now.
        // If this memory allocation was already captured for this kernel, it
        // means that kernel is submitted several times. Increase reference
        // count only once because we release all allocations only when
        // SubmissionsCount turns to 0. We don't want to know how many times
        // allocation was retained by each submission.
        if (Pair.second)
          Elem.second.RefCount.increment();
      }
    }
    Kernel->SubmissionsCount++;
  }
  KernelsToBeSubmitted.clear();
}

pi_result _pi_queue::executeCommandList(pi_command_list_ptr_t CommandList,
                                        bool IsBlocking,
                                        bool OKToBatchCommand) {
  // Do nothing if command list is already closed.
  if (CommandList->second.IsClosed)
    return PI_SUCCESS;

  bool UseCopyEngine = CommandList->second.isCopy(this);

  // If the current LastCommandEvent is the nullptr, then it means
  // either that no command has ever been issued to the queue
  // or it means that the LastCommandEvent has been signalled and
  // therefore that this Queue is idle.
  //
  // NOTE: this behavior adds some flakyness to the batching
  // since last command's event may or may not be completed by the
  // time we get here depending on timings and system/gpu load.
  // So, disable it for modes where we print PI traces. Printing
  // traces incurs much different timings than real execution
  // ansyway, and many regression tests use it.
  //
  bool CurrentlyEmpty = !PrintTrace && this->LastCommandEvent == nullptr;

  // The list can be empty if command-list only contains signals of proxy
  // events. It is possible that executeCommandList is called twice for the same
  // command list without new appended command. We don't to want process the
  // same last command event twice that's why additionally check that new
  // command was appended to the command list.
  if (!CommandList->second.EventList.empty() &&
      this->LastCommandEvent != CommandList->second.EventList.back()) {
    this->LastCommandEvent = CommandList->second.EventList.back();
    if (doReuseDiscardedEvents()) {
      PI_CALL(resetDiscardedEvent(CommandList));
    }
  }

  this->LastUsedCommandList = CommandList;

  if (!UsingImmCmdLists) {
    // Batch if allowed to, but don't batch if we know there are no kernels
    // from this queue that are currently executing.  This is intended to get
    // kernels started as soon as possible when there are no kernels from this
    // queue awaiting execution, while allowing batching to occur when there
    // are kernels already executing. Also, if we are using fixed size batching,
    // as indicated by !ZeCommandListBatch.dynamic(), then just ignore
    // CurrentlyEmpty as we want to strictly follow the batching the user
    // specified.
    auto &CommandBatch = UseCopyEngine ? CopyCommandBatch : ComputeCommandBatch;
    auto &ZeCommandListBatchConfig = UseCopyEngine
                                         ? ZeCommandListBatchCopyConfig
                                         : ZeCommandListBatchComputeConfig;
    if (OKToBatchCommand && this->isBatchingAllowed(UseCopyEngine) &&
        (!ZeCommandListBatchConfig.dynamic() || !CurrentlyEmpty)) {

      if (hasOpenCommandList(UseCopyEngine) &&
          CommandBatch.OpenCommandList != CommandList)
        die("executeCommandList: OpenCommandList should be equal to"
            "null or CommandList");

      if (CommandList->second.size() < CommandBatch.QueueBatchSize) {
        CommandBatch.OpenCommandList = CommandList;
        return PI_SUCCESS;
      }

      adjustBatchSizeForFullBatch(UseCopyEngine);
      CommandBatch.OpenCommandList = CommandListMap.end();
    }
  }

  auto &ZeCommandQueue = CommandList->second.ZeQueue;
  // Scope of the lock must be till the end of the function, otherwise new mem
  // allocs can be created between the moment when we made a snapshot and the
  // moment when command list is closed and executed. But mutex is locked only
  // if indirect access tracking enabled, because std::defer_lock is used.
  // unique_lock destructor at the end of the function will unlock the mutex
  // if it was locked (which happens only if IndirectAccessTrackingEnabled is
  // true).
  std::unique_lock<ur_shared_mutex> ContextsLock(
      Device->Platform->ContextsMutex, std::defer_lock);

  if (IndirectAccessTrackingEnabled) {
    // We are going to submit kernels for execution. If indirect access flag is
    // set for a kernel then we need to make a snapshot of existing memory
    // allocations in all contexts in the platform. We need to lock the mutex
    // guarding the list of contexts in the platform to prevent creation of new
    // memory alocations in any context before we submit the kernel for
    // execution.
    ContextsLock.lock();
    CaptureIndirectAccesses();
  }

  if (!UsingImmCmdLists) {
    // In this mode all inner-batch events have device visibility only,
    // and we want the last command in the batch to signal a host-visible
    // event that anybody waiting for any event in the batch will
    // really be using.
    // We need to create a proxy host-visible event only if the list of events
    // in the command list is not empty, otherwise we are going to just create
    // and remove proxy event right away and dereference deleted object
    // afterwards.
    if (Device->ZeEventsScope == LastCommandInBatchHostVisible &&
        !CommandList->second.EventList.empty()) {
      // If there are only internal events in the command list then we don't
      // need to create host proxy event.
      auto Result =
          std::find_if(CommandList->second.EventList.begin(),
                       CommandList->second.EventList.end(),
                       [](pi_event E) { return E->hasExternalRefs(); });
      if (Result != CommandList->second.EventList.end()) {
        // Create a "proxy" host-visible event.
        //
        pi_event HostVisibleEvent;
        auto Res = createEventAndAssociateQueue(
            this, &HostVisibleEvent, PI_COMMAND_TYPE_USER, CommandList,
            /* IsInternal */ false, /* HostVisible */ true);
        if (Res)
          return Res;

        // Update each command's event in the command-list to "see" this
        // proxy event as a host-visible counterpart.
        for (auto &Event : CommandList->second.EventList) {
          std::scoped_lock<ur_shared_mutex> EventLock(Event->Mutex);
          // Internal event doesn't need host-visible proxy.
          if (!Event->hasExternalRefs())
            continue;

          if (!Event->HostVisibleEvent) {
            Event->HostVisibleEvent = HostVisibleEvent;
            HostVisibleEvent->RefCount.increment();
          }
        }

        // Decrement the reference count of the event such that all the
        // remaining references are from the other commands in this batch and
        // from the command-list itself. This host-visible event will not be
        // waited/released by SYCL RT, so it must be destroyed after all events
        // in the batch are gone. We know that refcount is more than 2 because
        // we check that EventList of the command list is not empty above, i.e.
        // after createEventAndAssociateQueue ref count is 2 and then +1 for
        // each event in the EventList.
        PI_CALL(piEventReleaseInternal(HostVisibleEvent));

        if (doReuseDiscardedEvents()) {
          // If we have in-order queue with discarded events then we want to
          // treat this event as regular event. We insert a barrier in the next
          // command list to wait for this event.
          LastCommandEvent = HostVisibleEvent;
        } else {
          // For all other queues treat this as a special event and indicate no
          // cleanup is needed.
          // TODO: always treat this host event as a regular event.
          PI_CALL(piEventReleaseInternal(HostVisibleEvent));
          HostVisibleEvent->CleanedUp = true;
        }

        // Finally set to signal the host-visible event at the end of the
        // command-list after a barrier that waits for all commands
        // completion.
        if (doReuseDiscardedEvents() && LastCommandEvent &&
            LastCommandEvent->IsDiscarded) {
          // If we the last event is discarded then we already have a barrier
          // inserted, so just signal the event.
          ZE_CALL(zeCommandListAppendSignalEvent,
                  (CommandList->first, HostVisibleEvent->ZeEvent));
        } else {
          ZE_CALL(zeCommandListAppendBarrier,
                  (CommandList->first, HostVisibleEvent->ZeEvent, 0, nullptr));
        }
      } else {
        // If we don't have host visible proxy then signal event if needed.
        this->signalEventFromCmdListIfLastEventDiscarded(CommandList);
      }
    } else {
      // If we don't have host visible proxy then signal event if needed.
      this->signalEventFromCmdListIfLastEventDiscarded(CommandList);
    }

    // Close the command list and have it ready for dispatch.
    ZE_CALL(zeCommandListClose, (CommandList->first));
    // Mark this command list as closed.
    CommandList->second.IsClosed = true;
    this->LastUsedCommandList = CommandListMap.end();
    // Offload command list to the GPU for asynchronous execution
    auto ZeCommandList = CommandList->first;
    auto ZeResult = ZE_CALL_NOCHECK(
        zeCommandQueueExecuteCommandLists,
        (ZeCommandQueue, 1, &ZeCommandList, CommandList->second.ZeFence));
    if (ZeResult != ZE_RESULT_SUCCESS) {
      this->Healthy = false;
      if (ZeResult == ZE_RESULT_ERROR_UNKNOWN) {
        // Turn into a more informative end-user error.
        return PI_ERROR_COMMAND_EXECUTION_FAILURE;
      }
      return mapError(ZeResult);
    }
  }

  // Check global control to make every command blocking for debugging.
  if (IsBlocking || (UrL0Serialize & UrL0SerializeBlock) != 0) {
    if (UsingImmCmdLists) {
      synchronize();
    } else {
      // Wait until command lists attached to the command queue are executed.
      ZE_CALL(zeHostSynchronize, (ZeCommandQueue));
    }
  }
  return PI_SUCCESS;
}

bool _pi_queue::isBatchingAllowed(bool IsCopy) const {
  auto &CommandBatch = IsCopy ? CopyCommandBatch : ComputeCommandBatch;
  return (CommandBatch.QueueBatchSize > 0 &&
          ((UrL0Serialize & UrL0SerializeBlock) == 0));
}

// Return the index of the next queue to use based on a
// round robin strategy and the queue group ordinal.
uint32_t _pi_queue::pi_queue_group_t::getQueueIndex(uint32_t *QueueGroupOrdinal,
                                                    uint32_t *QueueIndex,
                                                    bool QueryOnly) {
  auto CurrentIndex = NextIndex;

  if (!QueryOnly) {
    ++NextIndex;
    if (NextIndex > UpperIndex)
      NextIndex = LowerIndex;
  }

  // Find out the right queue group ordinal (first queue might be "main" or
  // "link")
  auto QueueType = Type;
  if (QueueType != queue_type::Compute)
    QueueType = (CurrentIndex == 0 && Queue->Device->hasMainCopyEngine())
                    ? queue_type::MainCopy
                    : queue_type::LinkCopy;

  *QueueGroupOrdinal = Queue->Device->QueueGroup[QueueType].ZeOrdinal;
  // Adjust the index to the L0 queue group since we represent "main" and
  // "link"
  // L0 groups with a single copy group ("main" would take "0" index).
  auto ZeCommandQueueIndex = CurrentIndex;
  if (QueueType == queue_type::LinkCopy && Queue->Device->hasMainCopyEngine()) {
    ZeCommandQueueIndex -= 1;
  }
  *QueueIndex = ZeCommandQueueIndex;

  return CurrentIndex;
}

int32_t _pi_queue::pi_queue_group_t::getCmdQueueOrdinal(
    ze_command_queue_handle_t CmdQueue) {
  // Find out the right queue group ordinal (first queue might be "main" or
  // "link")
  auto QueueType = Type;
  if (QueueType != queue_type::Compute)
    QueueType = (ZeQueues[0] == CmdQueue && Queue->Device->hasMainCopyEngine())
                    ? queue_type::MainCopy
                    : queue_type::LinkCopy;
  return Queue->Device->QueueGroup[QueueType].ZeOrdinal;
}

// This function will return one of possibly multiple available native
// queues and the value of the queue group ordinal.
ze_command_queue_handle_t &
_pi_queue::pi_queue_group_t::getZeQueue(uint32_t *QueueGroupOrdinal) {

  // QueueIndex is the proper L0 index.
  // Index is the plugins concept of index, with main and link copy engines in
  // one range.
  uint32_t QueueIndex;
  auto Index = getQueueIndex(QueueGroupOrdinal, &QueueIndex);

  ze_command_queue_handle_t &ZeQueue = ZeQueues[Index];
  if (ZeQueue)
    return ZeQueue;

  ZeStruct<ze_command_queue_desc_t> ZeCommandQueueDesc;
  ZeCommandQueueDesc.ordinal = *QueueGroupOrdinal;
  ZeCommandQueueDesc.index = QueueIndex;
  ZeCommandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  const char *Priority = "Normal";
  if (Queue->isPriorityLow()) {
    ZeCommandQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
    Priority = "Low";
  } else if (Queue->isPriorityHigh()) {
    ZeCommandQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
    Priority = "High";
  }

  // Evaluate performance of explicit usage for "0" index.
  if (QueueIndex != 0) {
    ZeCommandQueueDesc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;
  }

  urPrint("[getZeQueue]: create queue ordinal = %d, index = %d "
          "(round robin in [%d, %d]) priority = %s\n",
          ZeCommandQueueDesc.ordinal, ZeCommandQueueDesc.index, LowerIndex,
          UpperIndex, Priority);

  auto ZeResult = ZE_CALL_NOCHECK(
      zeCommandQueueCreate, (Queue->Context->ZeContext, Queue->Device->ZeDevice,
                             &ZeCommandQueueDesc, &ZeQueue));
  if (ZeResult) {
    die("[L0] getZeQueue: failed to create queue");
  }

  return ZeQueue;
}

// This function will return one of possibly multiple available
// immediate commandlists associated with this Queue.
pi_command_list_ptr_t &_pi_queue::pi_queue_group_t::getImmCmdList() {

  uint32_t QueueIndex, QueueOrdinal;
  auto Index = getQueueIndex(&QueueOrdinal, &QueueIndex);

  if (ImmCmdLists[Index] != Queue->CommandListMap.end())
    return ImmCmdLists[Index];

  ZeStruct<ze_command_queue_desc_t> ZeCommandQueueDesc;
  ZeCommandQueueDesc.ordinal = QueueOrdinal;
  ZeCommandQueueDesc.index = QueueIndex;
  ZeCommandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  const char *Priority = "Normal";
  if (Queue->isPriorityLow()) {
    ZeCommandQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
    Priority = "Low";
  } else if (Queue->isPriorityHigh()) {
    ZeCommandQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
    Priority = "High";
  }

  // Evaluate performance of explicit usage for "0" index.
  if (QueueIndex != 0) {
    ZeCommandQueueDesc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;
  }

  urPrint("[getZeQueue]: create queue ordinal = %d, index = %d "
          "(round robin in [%d, %d]) priority = %s\n",
          ZeCommandQueueDesc.ordinal, ZeCommandQueueDesc.index, LowerIndex,
          UpperIndex, Priority);

  ze_command_list_handle_t ZeCommandList;
  ZE_CALL_NOCHECK(zeCommandListCreateImmediate,
                  (Queue->Context->ZeContext, Queue->Device->ZeDevice,
                   &ZeCommandQueueDesc, &ZeCommandList));
  ImmCmdLists[Index] =
      Queue->CommandListMap
          .insert(std::pair<ze_command_list_handle_t, pi_command_list_info_t>{
              ZeCommandList, {nullptr, true, false, nullptr, QueueOrdinal}})
          .first;
  // Add this commandlist to the cache so it can be destroyed as part of
  // piQueueReleaseInternal
  auto QueueType = Type;
  std::scoped_lock<ur_mutex> Lock(Queue->Context->ZeCommandListCacheMutex);
  auto &ZeCommandListCache =
      QueueType == queue_type::Compute
          ? Queue->Context->ZeComputeCommandListCache[Queue->Device->ZeDevice]
          : Queue->Context->ZeCopyCommandListCache[Queue->Device->ZeDevice];
  ZeCommandListCache.push_back(ZeCommandList);

  return ImmCmdLists[Index];
}

pi_command_list_ptr_t _pi_queue::eventOpenCommandList(pi_event Event) {
  using IsCopy = bool;

  if (UsingImmCmdLists) {
    // When using immediate commandlists there are no open command lists.
    return CommandListMap.end();
  }

  if (hasOpenCommandList(IsCopy{false})) {
    const auto &ComputeEventList =
        ComputeCommandBatch.OpenCommandList->second.EventList;
    if (std::find(ComputeEventList.begin(), ComputeEventList.end(), Event) !=
        ComputeEventList.end())
      return ComputeCommandBatch.OpenCommandList;
  }
  if (hasOpenCommandList(IsCopy{true})) {
    const auto &CopyEventList =
        CopyCommandBatch.OpenCommandList->second.EventList;
    if (std::find(CopyEventList.begin(), CopyEventList.end(), Event) !=
        CopyEventList.end())
      return CopyCommandBatch.OpenCommandList;
  }
  return CommandListMap.end();
}

pi_result _pi_queue::insertStartBarrierIfDiscardEventsMode(
    pi_command_list_ptr_t &CmdList) {
  // If current command list is different from the last command list then insert
  // a barrier waiting for the last command event.
  if (doReuseDiscardedEvents() && CmdList != LastUsedCommandList &&
      LastCommandEvent) {
    ZE_CALL(zeCommandListAppendBarrier,
            (CmdList->first, nullptr, 1, &(LastCommandEvent->ZeEvent)));
    LastCommandEvent = nullptr;
  }
  return PI_SUCCESS;
}

pi_result _pi_queue::insertActiveBarriers(pi_command_list_ptr_t &CmdList,
                                          bool UseCopyEngine) {
  // Early exit if there are no active barriers.
  if (ActiveBarriers.empty())
    return PI_SUCCESS;

  // Create a wait-list and retain events.
  _pi_ze_event_list_t ActiveBarriersWaitList;
  if (auto Res = ActiveBarriersWaitList.createAndRetainPiZeEventList(
          ActiveBarriers.vector().size(), ActiveBarriers.vector().data(), this,
          UseCopyEngine))
    return Res;

  // We can now replace active barriers with the ones in the wait list.
  if (auto Res = ActiveBarriers.clear())
    return Res;

  if (ActiveBarriersWaitList.Length == 0) {
    return PI_SUCCESS;
  }

  for (pi_uint32 I = 0; I < ActiveBarriersWaitList.Length; ++I) {
    auto &Event = ActiveBarriersWaitList.PiEventList[I];
    ActiveBarriers.add(Event);
  }

  pi_event Event = nullptr;
  if (auto Res = createEventAndAssociateQueue(
          this, &Event, PI_COMMAND_TYPE_USER, CmdList, /*IsInternal*/ true))
    return Res;

  Event->WaitList = ActiveBarriersWaitList;
  Event->OwnZeEvent = true;

  // If there are more active barriers, insert a barrier on the command-list. We
  // do not need an event for finishing so we pass nullptr.
  ZE_CALL(zeCommandListAppendBarrier,
          (CmdList->first, nullptr, ActiveBarriersWaitList.Length,
           ActiveBarriersWaitList.ZeEventList));
  return PI_SUCCESS;
}

pi_result _pi_queue::executeOpenCommandList(bool IsCopy) {
  auto &CommandBatch = IsCopy ? CopyCommandBatch : ComputeCommandBatch;
  // If there are any commands still in the open command list for this
  // queue, then close and execute that command list now.
  if (hasOpenCommandList(IsCopy)) {
    adjustBatchSizeForPartialBatch(IsCopy);
    auto Res = executeCommandList(CommandBatch.OpenCommandList, false, false);
    CommandBatch.OpenCommandList = CommandListMap.end();
    return Res;
  }

  return PI_SUCCESS;
}

static const bool FilterEventWaitList = [] {
  const char *UrRet = std::getenv("UR_L0_FILTER_EVENT_WAIT_LIST");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_FILTER_EVENT_WAIT_LIST");
  return (UrRet ? std::stoi(UrRet) : (PiRet ? std::stoi(PiRet) : 0));
}();

pi_result _pi_ze_event_list_t::createAndRetainPiZeEventList(
    pi_uint32 EventListLength, const pi_event *EventList, pi_queue CurQueue,
    bool UseCopyEngine) {
  this->Length = 0;
  this->ZeEventList = nullptr;
  this->PiEventList = nullptr;

  if (CurQueue->isInOrderQueue() && CurQueue->LastCommandEvent != nullptr) {
    if (CurQueue->UsingImmCmdLists) {
      if (ReuseDiscardedEvents && CurQueue->isDiscardEvents()) {
        // If queue is in-order with discarded events and if
        // new command list is different from the last used command list then
        // signal new event from the last immediate command list. We are going
        // to insert a barrier in the new command list waiting for that event.
        auto QueueGroup = CurQueue->getQueueGroup(UseCopyEngine);
        uint32_t QueueGroupOrdinal, QueueIndex;
        auto NextIndex =
            QueueGroup.getQueueIndex(&QueueGroupOrdinal, &QueueIndex,
                                     /*QueryOnly */ true);
        auto NextImmCmdList = QueueGroup.ImmCmdLists[NextIndex];
        if (CurQueue->LastUsedCommandList != CurQueue->CommandListMap.end() &&
            CurQueue->LastUsedCommandList != NextImmCmdList) {
          CurQueue->signalEventFromCmdListIfLastEventDiscarded(
              CurQueue->LastUsedCommandList);
        }
      }
    } else {
      // Ensure LastCommandEvent's batch is submitted if it is differrent
      // from the one this command is going to. If we reuse discarded events
      // then signalEventFromCmdListIfLastEventDiscarded will be called at batch
      // close if needed.
      const auto &OpenCommandList =
          CurQueue->eventOpenCommandList(CurQueue->LastCommandEvent);
      if (OpenCommandList != CurQueue->CommandListMap.end() &&
          OpenCommandList->second.isCopy(CurQueue) != UseCopyEngine) {

        if (auto Res = CurQueue->executeOpenCommandList(
                OpenCommandList->second.isCopy(CurQueue)))
          return Res;
      }
    }
  }

  // For in-order queues, every command should be executed only after the
  // previous command has finished. The event associated with the last
  // enqueued command is added into the waitlist to ensure in-order semantics.
  bool IncludeLastCommandEvent =
      CurQueue->isInOrderQueue() && CurQueue->LastCommandEvent != nullptr;

  // If the last event is discarded then we already have a barrier waiting for
  // that event, so must not include the last command event into the wait
  // list because it will cause waiting for event which was reset.
  if (ReuseDiscardedEvents && CurQueue->isDiscardEvents() &&
      CurQueue->LastCommandEvent && CurQueue->LastCommandEvent->IsDiscarded)
    IncludeLastCommandEvent = false;

  try {
    pi_uint32 TmpListLength = 0;

    if (IncludeLastCommandEvent) {
      this->ZeEventList = new ze_event_handle_t[EventListLength + 1];
      this->PiEventList = new pi_event[EventListLength + 1];
      std::shared_lock<ur_shared_mutex> Lock(CurQueue->LastCommandEvent->Mutex);
      this->ZeEventList[0] = CurQueue->LastCommandEvent->ZeEvent;
      this->PiEventList[0] = CurQueue->LastCommandEvent;
      TmpListLength = 1;
    } else if (EventListLength > 0) {
      this->ZeEventList = new ze_event_handle_t[EventListLength];
      this->PiEventList = new pi_event[EventListLength];
    }

    if (EventListLength > 0) {
      for (pi_uint32 I = 0; I < EventListLength; I++) {
        PI_ASSERT(EventList[I] != nullptr, PI_ERROR_INVALID_VALUE);
        {
          std::shared_lock<ur_shared_mutex> Lock(EventList[I]->Mutex);
          if (EventList[I]->Completed)
            continue;

          // Poll of the host-visible events.
          auto HostVisibleEvent = EventList[I]->HostVisibleEvent;
          if (FilterEventWaitList && HostVisibleEvent) {
            auto Res = ZE_CALL_NOCHECK(zeEventQueryStatus,
                                       (HostVisibleEvent->ZeEvent));
            if (Res == ZE_RESULT_SUCCESS) {
              // Event has already completed, don't put it into the list
              continue;
            }
          }
        }

        auto Queue = EventList[I]->Queue;
        if (Queue) {
          // The caller of createAndRetainPiZeEventList must already hold
          // a lock of the CurQueue. Additionally lock the Queue if it
          // is different from CurQueue.
          // TODO: rework this to avoid deadlock when another thread is
          //       locking the same queues but in a different order.
          auto Lock = ((Queue == CurQueue)
                           ? std::unique_lock<ur_shared_mutex>()
                           : std::unique_lock<ur_shared_mutex>(Queue->Mutex));

          // If the event that is going to be waited is in an open batch
          // different from where this next command is going to be added,
          // then we have to force execute of that open command-list
          // to avoid deadlocks.
          //
          const auto &OpenCommandList =
              Queue->eventOpenCommandList(EventList[I]);
          if (OpenCommandList != Queue->CommandListMap.end()) {

            if (Queue == CurQueue &&
                OpenCommandList->second.isCopy(Queue) == UseCopyEngine) {
              // Don't force execute the batch yet since the new command
              // is going to the same open batch as the dependent event.
            } else {
              if (auto Res = Queue->executeOpenCommandList(
                      OpenCommandList->second.isCopy(Queue)))
                return Res;
            }
          }
        } else {
          // There is a dependency on an interop-event.
          // Similarily to the above to avoid dead locks ensure that
          // execution of all prior commands in the current command-
          // batch is visible to the host. This may not be the case
          // when we intended to have only last command in the batch
          // produce host-visible event, e.g.
          //
          //  event0 = interop event
          //  event1 = command1 (already in batch, no deps)
          //  event2 = command2 (is being added, dep on event0)
          //  event3 = signal host-visible event for the batch
          //  event1.wait()
          //  event0.signal()
          //
          // Make sure that event1.wait() will wait for a host-visible
          // event that is signalled before the command2 is enqueued.
          if (CurQueue->Device->ZeEventsScope != AllHostVisible) {
            CurQueue->executeAllOpenCommandLists();
          }
        }

        std::shared_lock<ur_shared_mutex> Lock(EventList[I]->Mutex);
        this->ZeEventList[TmpListLength] = EventList[I]->ZeEvent;
        this->PiEventList[TmpListLength] = EventList[I];
        TmpListLength += 1;
      }
    }

    this->Length = TmpListLength;

  } catch (...) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  }

  for (pi_uint32 I = 0; I < this->Length; I++) {
    this->PiEventList[I]->RefCount.increment();
  }

  return PI_SUCCESS;
}

static void printZeEventList(const _pi_ze_event_list_t &PiZeEventList) {
  urPrint("  NumEventsInWaitList %d:", PiZeEventList.Length);

  for (pi_uint32 I = 0; I < PiZeEventList.Length; I++) {
    urPrint(" %#llx", ur_cast<std::uintptr_t>(PiZeEventList.ZeEventList[I]));
  }

  urPrint("\n");
}

pi_result _pi_ze_event_list_t::collectEventsForReleaseAndDestroyPiZeEventList(
    std::list<pi_event> &EventsToBeReleased) {
  // acquire a lock before reading the length and list fields.
  // Acquire the lock, copy the needed data locally, and reset
  // the fields, then release the lock.
  // Only then do we do the actual actions to release and destroy,
  // holding the lock for the minimum time necessary.
  pi_uint32 LocLength = 0;
  ze_event_handle_t *LocZeEventList = nullptr;
  pi_event *LocPiEventList = nullptr;

  {
    // acquire the lock and copy fields locally
    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_mutex> lock(this->PiZeEventListMutex);

    LocLength = Length;
    LocZeEventList = ZeEventList;
    LocPiEventList = PiEventList;

    Length = 0;
    ZeEventList = nullptr;
    PiEventList = nullptr;

    // release lock by ending scope.
  }

  for (pi_uint32 I = 0; I < LocLength; I++) {
    // Add the event to be released to the list
    EventsToBeReleased.push_back(LocPiEventList[I]);
  }

  if (LocZeEventList != nullptr) {
    delete[] LocZeEventList;
  }
  if (LocPiEventList != nullptr) {
    delete[] LocPiEventList;
  }

  return PI_SUCCESS;
}

extern "C" {

// Forward declarations
decltype(piEventCreate) piEventCreate;

static ze_result_t
checkUnresolvedSymbols(ze_module_handle_t ZeModule,
                       ze_module_build_log_handle_t *ZeBuildLog);

pi_result piPlatformsGet(pi_uint32 NumEntries, pi_platform *Platforms,
                         pi_uint32 *NumPlatforms) {
  return pi2ur::piPlatformsGet(NumEntries, Platforms, NumPlatforms);
}

pi_result piPlatformGetInfo(pi_platform Platform, pi_platform_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {
  urPrint("==========================\n");
  urPrint("SYCL over Level-Zero %s\n", Platform->ZeDriverVersion.c_str());
  urPrint("==========================\n");

  // To distinguish this L0 platform from Unified Runtime one.
  if (ParamName == PI_PLATFORM_INFO_NAME) {
    ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
    return ReturnValue("Intel(R) Level-Zero");
  }
  return pi2ur::piPlatformGetInfo(Platform, ParamName, ParamValueSize,
                                  ParamValue, ParamValueSizeRet);
}

pi_result piextPlatformGetNativeHandle(pi_platform Platform,
                                       pi_native_handle *NativeHandle) {
  PI_ASSERT(Platform, PI_ERROR_INVALID_PLATFORM);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  auto ZeDriver = ur_cast<ze_driver_handle_t *>(NativeHandle);
  // Extract the Level Zero driver handle from the given PI platform
  *ZeDriver = Platform->ZeDriver;
  return PI_SUCCESS;
}

pi_result piextPlatformCreateWithNativeHandle(pi_native_handle NativeHandle,
                                              pi_platform *Platform) {
  PI_ASSERT(Platform, PI_ERROR_INVALID_PLATFORM);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  auto ZeDriver = ur_cast<ze_driver_handle_t>(NativeHandle);

  pi_uint32 NumPlatforms = 0;
  pi_result Res = piPlatformsGet(0, nullptr, &NumPlatforms);
  if (Res != PI_SUCCESS) {
    return Res;
  }

  if (NumPlatforms) {
    std::vector<pi_platform> Platforms(NumPlatforms);
    PI_CALL(piPlatformsGet(NumPlatforms, Platforms.data(), nullptr));

    // The SYCL spec requires that the set of platforms must remain fixed for
    // the duration of the application's execution. We assume that we found all
    // of the Level Zero drivers when we initialized the platform cache, so the
    // "NativeHandle" must already be in the cache. If it is not, this must not
    // be a valid Level Zero driver.
    for (const pi_platform &CachedPlatform : Platforms) {
      if (CachedPlatform->ZeDriver == ZeDriver) {
        *Platform = CachedPlatform;
        return PI_SUCCESS;
      }
    }
  }

  return PI_ERROR_INVALID_VALUE;
}

pi_result piPluginGetLastError(char **message) {
  return pi2ur::piPluginGetLastError(message);
}

// Returns plugin specific backend option.
// Current support is only for optimization options.
// Return '-ze-opt-disable' for frontend_option = -O0.
// Return '-ze-opt-level=1' for frontend_option = -O1 or -O2.
// Return '-ze-opt-level=2' for frontend_option = -O3.
pi_result piPluginGetBackendOption(pi_platform, const char *frontend_option,
                                   const char **backend_option) {
  using namespace std::literals;
  if (frontend_option == nullptr) {
    return PI_ERROR_INVALID_VALUE;
  }
  if (frontend_option == ""sv) {
    *backend_option = "";
    return PI_SUCCESS;
  }
  if (frontend_option == "-O0"sv) {
    *backend_option = "-ze-opt-disable";
    return PI_SUCCESS;
  }
  if (frontend_option == "-O1"sv || frontend_option == "-O2"sv) {
    *backend_option = "-ze-opt-level=1";
    return PI_SUCCESS;
  }
  if (frontend_option == "-O3"sv) {
    *backend_option = "-ze-opt-level=2";
    return PI_SUCCESS;
  }
  return PI_ERROR_INVALID_VALUE;
}

pi_result piDevicesGet(pi_platform Platform, pi_device_type DeviceType,
                       pi_uint32 NumEntries, pi_device *Devices,
                       pi_uint32 *NumDevices) {
  return pi2ur::piDevicesGet(Platform, DeviceType, NumEntries, Devices,
                             NumDevices);
}

pi_result piDeviceRetain(pi_device Device) {
  return pi2ur::piDeviceRetain(Device);
}

pi_result piDeviceRelease(pi_device Device) {
  return pi2ur::piDeviceRelease(Device);
}

pi_result piDeviceGetInfo(pi_device Device, pi_device_info ParamName,
                          size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet) {
  return pi2ur::piDeviceGetInfo(Device, ParamName, ParamValueSize, ParamValue,
                                ParamValueSizeRet);
}

pi_result piDevicePartition(pi_device Device,
                            const pi_device_partition_property *Properties,
                            pi_uint32 NumDevices, pi_device *OutDevices,
                            pi_uint32 *OutNumDevices) {
  return pi2ur::piDevicePartition(Device, Properties, NumDevices, OutDevices,
                                  OutNumDevices);
}

pi_result
piextDeviceSelectBinary(pi_device Device, // TODO: does this need to be context?
                        pi_device_binary *Binaries, pi_uint32 NumBinaries,
                        pi_uint32 *SelectedBinaryInd) {

  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(SelectedBinaryInd, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(NumBinaries == 0 || Binaries, PI_ERROR_INVALID_VALUE);

  // TODO: this is a bare-bones implementation for choosing a device image
  // that would be compatible with the targeted device. An AOT-compiled
  // image is preferred over SPIR-V for known devices (i.e. Intel devices)
  // The implementation makes no effort to differentiate between multiple images
  // for the given device, and simply picks the first one compatible.
  //
  // Real implementation will use the same mechanism OpenCL ICD dispatcher
  // uses. Something like:
  //   PI_VALIDATE_HANDLE_RETURN_HANDLE(ctx, PI_ERROR_INVALID_CONTEXT);
  //     return context->dispatch->piextDeviceSelectIR(
  //       ctx, images, num_images, selected_image);
  // where context->dispatch is set to the dispatch table provided by PI
  // plugin for platform/device the ctx was created for.

  // Look for GEN binary, which we known can only be handled by Level-Zero now.
  const char *BinaryTarget = __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_GEN;

  // Find the appropriate device image, fallback to spirv if not found
  constexpr pi_uint32 InvalidInd = std::numeric_limits<pi_uint32>::max();
  pi_uint32 Spirv = InvalidInd;

  for (pi_uint32 i = 0; i < NumBinaries; ++i) {
    if (strcmp(Binaries[i]->DeviceTargetSpec, BinaryTarget) == 0) {
      *SelectedBinaryInd = i;
      return PI_SUCCESS;
    }
    if (strcmp(Binaries[i]->DeviceTargetSpec,
               __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64) == 0)
      Spirv = i;
  }
  // Points to a spirv image, if such indeed was found
  if ((*SelectedBinaryInd = Spirv) != InvalidInd)
    return PI_SUCCESS;

  // No image can be loaded for the given device
  return PI_ERROR_INVALID_BINARY;
}

pi_result piextDeviceGetNativeHandle(pi_device Device,
                                     pi_native_handle *NativeHandle) {
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  auto ZeDevice = ur_cast<ze_device_handle_t *>(NativeHandle);
  // Extract the Level Zero module handle from the given PI device
  *ZeDevice = Device->ZeDevice;
  return PI_SUCCESS;
}

pi_result piextDeviceCreateWithNativeHandle(pi_native_handle NativeHandle,
                                            pi_platform Platform,
                                            pi_device *Device) {
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  auto ZeDevice = ur_cast<ze_device_handle_t>(NativeHandle);

  // The SYCL spec requires that the set of devices must remain fixed for the
  // duration of the application's execution. We assume that we found all of the
  // Level Zero devices when we initialized the platforms/devices cache, so the
  // "NativeHandle" must already be in the cache. If it is not, this must not be
  // a valid Level Zero device.
  //
  // TODO: maybe we should populate cache of platforms if it wasn't already.
  // For now assert that is was populated.
  PI_ASSERT(PiPlatformCachePopulated, PI_ERROR_INVALID_VALUE);
  const std::lock_guard<SpinLock> Lock{*PiPlatformsCacheMutex};

  pi_device Dev = nullptr;
  for (pi_platform ThePlatform : *PiPlatformsCache) {
    Dev = ThePlatform->getDeviceFromNativeHandle(ZeDevice);
    if (Dev) {
      // Check that the input Platform, if was given, matches the found one.
      PI_ASSERT(!Platform || Platform == ThePlatform,
                PI_ERROR_INVALID_PLATFORM);
      break;
    }
  }

  if (Dev == nullptr)
    return PI_ERROR_INVALID_VALUE;

  *Device = Dev;
  return PI_SUCCESS;
}

pi_result piContextCreate(const pi_context_properties *Properties,
                          pi_uint32 NumDevices, const pi_device *Devices,
                          void (*PFnNotify)(const char *ErrInfo,
                                            const void *PrivateInfo, size_t CB,
                                            void *UserData),
                          void *UserData, pi_context *RetContext) {
  (void)Properties;
  (void)PFnNotify;
  (void)UserData;
  PI_ASSERT(NumDevices, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Devices, PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(RetContext, PI_ERROR_INVALID_VALUE);

  pi_platform Platform = (*Devices)->Platform;
  ZeStruct<ze_context_desc_t> ContextDesc;
  ContextDesc.flags = 0;

  ze_context_handle_t ZeContext;
  ZE_CALL(zeContextCreate, (Platform->ZeDriver, &ContextDesc, &ZeContext));
  try {
    *RetContext = new _pi_context(ZeContext, NumDevices, Devices, true);
    (*RetContext)->initialize();
    if (IndirectAccessTrackingEnabled) {
      std::scoped_lock<ur_shared_mutex> Lock(Platform->ContextsMutex);
      Platform->Contexts.push_back(*RetContext);
    }
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piContextGetInfo(pi_context Context, pi_context_info ParamName,
                           size_t ParamValueSize, void *ParamValue,
                           size_t *ParamValueSizeRet) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  std::shared_lock<ur_shared_mutex> Lock(Context->Mutex);
  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_CONTEXT_INFO_DEVICES:
    return ReturnValue(&Context->Devices[0], Context->Devices.size());
  case PI_CONTEXT_INFO_NUM_DEVICES:
    return ReturnValue(pi_uint32(Context->Devices.size()));
  case PI_CONTEXT_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Context->RefCount.load()});
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT:
    // 2D USM memcpy is supported unless disabled through
    // UR_L0_LEVEL_ZERO_USE_NATIVE_USM_MEMCPY2D.
    return ReturnValue(pi_bool{UseMemcpy2DOperations});
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT:
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMSET2D_SUPPORT:
    // 2D USM fill and memset is not supported.
    return ReturnValue(pi_bool{false});
  case PI_EXT_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES:
  case PI_EXT_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  case PI_EXT_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES:
  case PI_EXT_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES: {
    // These queries should be dealt with in context_impl.cpp by calling the
    // queries of each device separately and building the intersection set.
    setErrorMessage("These queries should have never come here.",
                    UR_RESULT_ERROR_INVALID_VALUE);
    return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
  }
  default:
    // TODO: implement other parameters
    die("piGetContextInfo: unsuppported ParamName.");
  }

  return PI_SUCCESS;
}

// FIXME: Dummy implementation to prevent link fail
pi_result piextContextSetExtendedDeleter(pi_context Context,
                                         pi_context_extended_deleter Function,
                                         void *UserData) {
  (void)Context;
  (void)Function;
  (void)UserData;
  die("piextContextSetExtendedDeleter: not supported");
  return PI_SUCCESS;
}

pi_result piextContextGetNativeHandle(pi_context Context,
                                      pi_native_handle *NativeHandle) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  auto ZeContext = ur_cast<ze_context_handle_t *>(NativeHandle);
  // Extract the Level Zero queue handle from the given PI queue
  *ZeContext = Context->ZeContext;
  return PI_SUCCESS;
}

pi_result piextContextCreateWithNativeHandle(pi_native_handle NativeHandle,
                                             pi_uint32 NumDevices,
                                             const pi_device *Devices,
                                             bool OwnNativeHandle,
                                             pi_context *RetContext) {
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Devices, PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(RetContext, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(NumDevices, PI_ERROR_INVALID_VALUE);

  try {
    *RetContext = new _pi_context(ur_cast<ze_context_handle_t>(NativeHandle),
                                  NumDevices, Devices, OwnNativeHandle);
    (*RetContext)->initialize();
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piContextRetain(pi_context Context) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  Context->RefCount.increment();
  return PI_SUCCESS;
}

// Helper function to release the context, a caller must lock the platform-level
// mutex guarding the container with contexts because the context can be removed
// from the list of tracked contexts.
pi_result ContextReleaseHelper(pi_context Context) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  if (!Context->RefCount.decrementAndTest())
    return PI_SUCCESS;

  if (IndirectAccessTrackingEnabled) {
    pi_platform Plt = Context->getPlatform();
    auto &Contexts = Plt->Contexts;
    auto It = std::find(Contexts.begin(), Contexts.end(), Context);
    if (It != Contexts.end())
      Contexts.erase(It);
  }
  ze_context_handle_t DestoryZeContext =
      Context->OwnZeContext ? Context->ZeContext : nullptr;

  // Clean up any live memory associated with Context
  pi_result Result = Context->finalize();

  // We must delete Context first and then destroy zeContext because
  // Context deallocation requires ZeContext in some member deallocation of
  // pi_context.
  delete Context;

  // Destruction of some members of pi_context uses L0 context
  // and therefore it must be valid at that point.
  // Technically it should be placed to the destructor of pi_context
  // but this makes API error handling more complex.
  if (DestoryZeContext) {
    auto ZeResult = ZE_CALL_NOCHECK(zeContextDestroy, (DestoryZeContext));
    // Gracefully handle the case that L0 was already unloaded.
    if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
      return mapError(ZeResult);
  }
  return Result;
}

pi_result piContextRelease(pi_context Context) {
  pi_platform Plt = Context->getPlatform();
  std::unique_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex,
                                                 std::defer_lock);
  if (IndirectAccessTrackingEnabled)
    ContextsLock.lock();

  return ContextReleaseHelper(Context);
}

pi_result piQueueCreate(pi_context Context, pi_device Device,
                        pi_queue_properties Flags, pi_queue *Queue) {
  pi_queue_properties Properties[] = {PI_QUEUE_FLAGS, Flags, 0};
  return piextQueueCreate(Context, Device, Properties, Queue);
}

pi_result piextQueueCreate(pi_context Context, pi_device Device,
                           pi_queue_properties *Properties, pi_queue *Queue) {
  PI_ASSERT(Properties, PI_ERROR_INVALID_VALUE);
  // Expect flags mask to be passed first.
  PI_ASSERT(Properties[0] == PI_QUEUE_FLAGS, PI_ERROR_INVALID_VALUE);
  pi_queue_properties Flags = Properties[1];

  PI_ASSERT(Properties[2] == 0 ||
                (Properties[2] == PI_QUEUE_COMPUTE_INDEX && Properties[4] == 0),
            PI_ERROR_INVALID_VALUE);
  auto ForceComputeIndex = Properties[2] == PI_QUEUE_COMPUTE_INDEX
                               ? static_cast<int>(Properties[3])
                               : -1; // Use default/round-robin.

  // Check that unexpected bits are not set.
  PI_ASSERT(
      !(Flags & ~(PI_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                  PI_QUEUE_FLAG_PROFILING_ENABLE | PI_QUEUE_FLAG_ON_DEVICE |
                  PI_QUEUE_FLAG_ON_DEVICE_DEFAULT |
                  PI_EXT_ONEAPI_QUEUE_FLAG_DISCARD_EVENTS |
                  PI_EXT_ONEAPI_QUEUE_FLAG_PRIORITY_LOW |
                  PI_EXT_ONEAPI_QUEUE_FLAG_PRIORITY_HIGH)),
      PI_ERROR_INVALID_VALUE);

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(Context->isValidDevice(Device), PI_ERROR_INVALID_DEVICE);

  // Create placeholder queues in the compute queue group.
  // Actual L0 queues will be created at first use.
  std::vector<ze_command_queue_handle_t> ZeComputeCommandQueues(
      Device->QueueGroup[_pi_queue::queue_type::Compute].ZeProperties.numQueues,
      nullptr);

  // Create placeholder queues in the copy queue group (main and link
  // native groups are combined into one group).
  // Actual L0 queues will be created at first use.
  size_t NumCopyGroups = 0;
  if (Device->hasMainCopyEngine()) {
    NumCopyGroups += Device->QueueGroup[_pi_queue::queue_type::MainCopy]
                         .ZeProperties.numQueues;
  }
  if (Device->hasLinkCopyEngine()) {
    NumCopyGroups += Device->QueueGroup[_pi_queue::queue_type::LinkCopy]
                         .ZeProperties.numQueues;
  }
  std::vector<ze_command_queue_handle_t> ZeCopyCommandQueues(NumCopyGroups,
                                                             nullptr);

  try {
    *Queue = new _pi_queue(ZeComputeCommandQueues, ZeCopyCommandQueues, Context,
                           Device, true, Flags, ForceComputeIndex);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  // Do eager initialization of Level Zero handles on request.
  if (doEagerInit) {
    pi_queue Q = *Queue;
    // Creates said number of command-lists.
    auto warmupQueueGroup = [Q](bool UseCopyEngine,
                                uint32_t RepeatCount) -> pi_result {
      pi_command_list_ptr_t CommandList;
      while (RepeatCount--) {
        if (Q->UsingImmCmdLists) {
          CommandList = Q->getQueueGroup(UseCopyEngine).getImmCmdList();
        } else {
          // Heuristically create some number of regular command-list to reuse.
          for (int I = 0; I < 10; ++I) {
            PI_CALL(Q->createCommandList(UseCopyEngine, CommandList));
            // Immediately return them to the cache of available command-lists.
            std::vector<pi_event> EventsUnused;
            PI_CALL(Q->resetCommandList(CommandList, true /* MakeAvailable */,
                                        EventsUnused));
          }
        }
      }
      return PI_SUCCESS;
    };
    // Create as many command-lists as there are queues in the group.
    // With this the underlying round-robin logic would initialize all
    // native queues, and create command-lists and their fences.
    // At this point only the thread creating the queue will have associated
    // command-lists. Other threads have not accessed the queue yet. So we can
    // only warmup the initial thread's command-lists.
    auto QueueGroup = Q->ComputeQueueGroupsByTID.get();
    PI_CALL(warmupQueueGroup(false, QueueGroup.UpperIndex -
                                        QueueGroup.LowerIndex + 1));
    if (Q->useCopyEngine()) {
      auto QueueGroup = Q->CopyQueueGroupsByTID.get();
      PI_CALL(warmupQueueGroup(true, QueueGroup.UpperIndex -
                                         QueueGroup.LowerIndex + 1));
    }
    // TODO: warmup event pools. Both host-visible and device-only.
  }
  return PI_SUCCESS;
}

pi_result piQueueGetInfo(pi_queue Queue, pi_queue_info ParamName,
                         size_t ParamValueSize, void *ParamValue,
                         size_t *ParamValueSizeRet) {

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  std::shared_lock<ur_shared_mutex> Lock(Queue->Mutex);
  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  // TODO: consider support for queue properties and size
  switch (ParamName) {
  case PI_QUEUE_INFO_CONTEXT:
    return ReturnValue(Queue->Context);
  case PI_QUEUE_INFO_DEVICE:
    return ReturnValue(Queue->Device);
  case PI_QUEUE_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Queue->RefCount.load()});
  case PI_QUEUE_INFO_PROPERTIES:
    die("PI_QUEUE_INFO_PROPERTIES in piQueueGetInfo not implemented\n");
    break;
  case PI_QUEUE_INFO_SIZE:
    die("PI_QUEUE_INFO_SIZE in piQueueGetInfo not implemented\n");
    break;
  case PI_QUEUE_INFO_DEVICE_DEFAULT:
    die("PI_QUEUE_INFO_DEVICE_DEFAULT in piQueueGetInfo not implemented\n");
    break;
  case PI_EXT_ONEAPI_QUEUE_INFO_EMPTY: {
    // We can exit early if we have in-order queue.
    if (Queue->isInOrderQueue()) {
      if (!Queue->LastCommandEvent)
        return ReturnValue(pi_bool{true});

      // We can check status of the event only if it isn't discarded otherwise
      // it may be reset (because we are free to reuse such events) and
      // zeEventQueryStatus will hang.
      // TODO: use more robust way to check that ZeEvent is not owned by
      // LastCommandEvent.
      if (!Queue->LastCommandEvent->IsDiscarded) {
        ze_result_t ZeResult = ZE_CALL_NOCHECK(
            zeEventQueryStatus, (Queue->LastCommandEvent->ZeEvent));
        if (ZeResult == ZE_RESULT_NOT_READY) {
          return ReturnValue(pi_bool{false});
        } else if (ZeResult != ZE_RESULT_SUCCESS) {
          return mapError(ZeResult);
        }
        return ReturnValue(pi_bool{true});
      }
      // For immediate command lists we have to check status of the event
      // because immediate command lists are not associated with level zero
      // queue. Conservatively return false in this case because last event is
      // discarded and we can't check its status.
      if (Queue->UsingImmCmdLists)
        return ReturnValue(pi_bool{false});
    }

    // If we have any open command list which is not empty then return false
    // because it means that there are commands which are not even submitted for
    // execution yet.
    using IsCopy = bool;
    if (Queue->hasOpenCommandList(IsCopy{true}) ||
        Queue->hasOpenCommandList(IsCopy{false}))
      return ReturnValue(pi_bool{false});

    for (const auto &QueueMap :
         {Queue->ComputeQueueGroupsByTID, Queue->CopyQueueGroupsByTID}) {
      for (const auto &QueueGroup : QueueMap) {
        if (Queue->UsingImmCmdLists) {
          // Immediate command lists are not associated with any Level Zero
          // queue, that's why we have to check status of events in each
          // immediate command list. Start checking from the end and exit early
          // if some event is not completed.
          for (const auto &ImmCmdList : QueueGroup.second.ImmCmdLists) {
            if (ImmCmdList == Queue->CommandListMap.end())
              continue;

            auto EventList = ImmCmdList->second.EventList;
            for (auto It = EventList.crbegin(); It != EventList.crend(); It++) {
              ze_result_t ZeResult =
                  ZE_CALL_NOCHECK(zeEventQueryStatus, ((*It)->ZeEvent));
              if (ZeResult == ZE_RESULT_NOT_READY) {
                return ReturnValue(pi_bool{false});
              } else if (ZeResult != ZE_RESULT_SUCCESS) {
                return mapError(ZeResult);
              }
            }
          }
        } else {
          for (const auto &ZeQueue : QueueGroup.second.ZeQueues) {
            if (!ZeQueue)
              continue;
            // Provide 0 as the timeout parameter to immediately get the status
            // of the Level Zero queue.
            ze_result_t ZeResult = ZE_CALL_NOCHECK(zeCommandQueueSynchronize,
                                                   (ZeQueue, /* timeout */ 0));
            if (ZeResult == ZE_RESULT_NOT_READY) {
              return ReturnValue(pi_bool{false});
            } else if (ZeResult != ZE_RESULT_SUCCESS) {
              return mapError(ZeResult);
            }
          }
        }
      }
    }
    return ReturnValue(pi_bool{true});
  }
  default:
    urPrint("Unsupported ParamName in piQueueGetInfo: ParamName=%d(0x%x)\n",
            ParamName, ParamName);
    return PI_ERROR_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

pi_result piQueueRetain(pi_queue Queue) {
  {
    std::scoped_lock<ur_shared_mutex> Lock(Queue->Mutex);
    Queue->RefCountExternal++;
  }
  Queue->RefCount.increment();
  return PI_SUCCESS;
}

pi_result piQueueRelease(pi_queue Queue) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);
  std::vector<pi_event> EventListToCleanup;

  {
    std::scoped_lock<ur_shared_mutex> Lock(Queue->Mutex);

    if ((--Queue->RefCountExternal) != 0)
      return PI_SUCCESS;

    // When external reference count goes to zero it is still possible
    // that internal references still exists, e.g. command-lists that
    // are not yet completed. So do full queue synchronization here
    // and perform proper cleanup.
    //
    // It is possible to get to here and still have an open command list
    // if no wait or finish ever occurred for this queue.
    if (auto Res = Queue->executeAllOpenCommandLists())
      return Res;

    // Make sure all commands get executed.
    Queue->synchronize();

    // Destroy all the fences created associated with this queue.
    for (auto it = Queue->CommandListMap.begin();
         it != Queue->CommandListMap.end(); ++it) {
      // This fence wasn't yet signalled when we polled it for recycling
      // the command-list, so need to release the command-list too.
      // For immediate commandlists we don't need to do an L0 reset of the
      // commandlist but do need to do event cleanup which is also in the
      // resetCommandList function.
      // If the fence is a nullptr we are using immediate commandlists,
      // otherwise regular commandlists which use a fence.
      if (it->second.ZeFence == nullptr || it->second.ZeFenceInUse) {
        Queue->resetCommandList(it, true, EventListToCleanup);
      }
      // TODO: remove "if" when the problem is fixed in the level zero
      // runtime. Destroy only if a queue is healthy. Destroying a fence may
      // cause a hang otherwise.
      // If the fence is a nullptr we are using immediate commandlists.
      if (Queue->Healthy && it->second.ZeFence != nullptr) {
        auto ZeResult = ZE_CALL_NOCHECK(zeFenceDestroy, (it->second.ZeFence));
        // Gracefully handle the case that L0 was already unloaded.
        if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
          return mapError(ZeResult);
      }
    }
    Queue->CommandListMap.clear();
  }

  for (auto &Event : EventListToCleanup) {
    // We don't need to synchronize the events since the queue
    // synchronized above already does that.
    {
      std::scoped_lock<ur_shared_mutex> EventLock(Event->Mutex);
      Event->Completed = true;
    }
    PI_CALL(CleanupCompletedEvent(Event));
    // This event was removed from the command list, so decrement ref count
    // (it was incremented when they were added to the command list).
    PI_CALL(piEventReleaseInternal(Event));
  }
  PI_CALL(piQueueReleaseInternal(Queue));
  return PI_SUCCESS;
}

static pi_result piQueueReleaseInternal(pi_queue Queue) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  if (!Queue->RefCount.decrementAndTest())
    return PI_SUCCESS;

  for (auto &Cache : Queue->EventCaches)
    for (auto &Event : Cache)
      PI_CALL(piEventReleaseInternal(Event));

  if (Queue->OwnZeCommandQueue) {
    for (auto &QueueMap :
         {Queue->ComputeQueueGroupsByTID, Queue->CopyQueueGroupsByTID})
      for (auto &QueueGroup : QueueMap)
        for (auto &ZeQueue : QueueGroup.second.ZeQueues)
          if (ZeQueue) {
            auto ZeResult = ZE_CALL_NOCHECK(zeCommandQueueDestroy, (ZeQueue));
            // Gracefully handle the case that L0 was already unloaded.
            if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
              return mapError(ZeResult);
          }
  }

  urPrint("piQueueRelease(compute) NumTimesClosedFull %d, "
          "NumTimesClosedEarly %d\n",
          Queue->ComputeCommandBatch.NumTimesClosedFull,
          Queue->ComputeCommandBatch.NumTimesClosedEarly);
  urPrint("piQueueRelease(copy) NumTimesClosedFull %d, NumTimesClosedEarly "
          "%d\n",
          Queue->CopyCommandBatch.NumTimesClosedFull,
          Queue->CopyCommandBatch.NumTimesClosedEarly);

  delete Queue;

  return PI_SUCCESS;
}

pi_result piQueueFinish(pi_queue Queue) {
  // Wait until command lists attached to the command queue are executed.
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  if (Queue->UsingImmCmdLists) {
    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> Lock(Queue->Mutex);

    Queue->synchronize();
  } else {
    std::unique_lock<ur_shared_mutex> Lock(Queue->Mutex);
    std::vector<ze_command_queue_handle_t> ZeQueues;

    // execute any command list that may still be open.
    if (auto Res = Queue->executeAllOpenCommandLists())
      return Res;

    // Make a copy of queues to sync and release the lock.
    for (auto &QueueMap :
         {Queue->ComputeQueueGroupsByTID, Queue->CopyQueueGroupsByTID})
      for (auto &QueueGroup : QueueMap)
        std::copy(QueueGroup.second.ZeQueues.begin(),
                  QueueGroup.second.ZeQueues.end(),
                  std::back_inserter(ZeQueues));

    // Remember the last command's event.
    auto LastCommandEvent = Queue->LastCommandEvent;

    // Don't hold a lock to the queue's mutex while waiting.
    // This allows continue working with the queue from other threads.
    // TODO: this currently exhibits some issues in the driver, so
    // we control this with an env var. Remove this control when
    // we settle one way or the other.
    const char *UrRet = std::getenv("UR_L0_QUEUE_FINISH_HOLD_LOCK");
    const char *PiRet =
        std::getenv("SYCL_PI_LEVEL_ZERO_QUEUE_FINISH_HOLD_LOCK");
    const bool HoldLock =
        UrRet ? std::stoi(UrRet) : (PiRet ? std::stoi(PiRet) : 0);

    if (!HoldLock) {
      Lock.unlock();
    }

    for (auto &ZeQueue : ZeQueues) {
      if (ZeQueue)
        ZE_CALL(zeHostSynchronize, (ZeQueue));
    }

    // Prevent unneeded already finished events to show up in the wait list.
    // We can only do so if nothing else was submitted to the queue
    // while we were synchronizing it.
    if (!HoldLock) {
      std::scoped_lock<ur_shared_mutex> Lock(Queue->Mutex);
      if (LastCommandEvent == Queue->LastCommandEvent) {
        Queue->LastCommandEvent = nullptr;
      }
    } else {
      Queue->LastCommandEvent = nullptr;
    }
  }
  // Reset signalled command lists and return them back to the cache of
  // available command lists. Events in the immediate command lists are cleaned
  // up in synchronize().
  if (!Queue->UsingImmCmdLists) {
    std::unique_lock<ur_shared_mutex> Lock(Queue->Mutex);
    resetCommandLists(Queue);
  }
  return PI_SUCCESS;
}

// Flushing cross-queue dependencies is covered by createAndRetainPiZeEventList,
// so this can be left as a no-op.
pi_result piQueueFlush(pi_queue Queue) {
  (void)Queue;
  return PI_SUCCESS;
}

pi_result piextQueueGetNativeHandle(pi_queue Queue,
                                    pi_native_handle *NativeHandle,
                                    int32_t *NativeHandleDesc) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(NativeHandleDesc, PI_ERROR_INVALID_VALUE);

  // Lock automatically releases when this goes out of scope.
  std::shared_lock<ur_shared_mutex> lock(Queue->Mutex);

  // Get handle to this thread's queue group.
  auto &QueueGroup = Queue->getQueueGroup(false /*compute*/);

  if (Queue->UsingImmCmdLists) {
    auto ZeCmdList = ur_cast<ze_command_list_handle_t *>(NativeHandle);
    // Extract the Level Zero command list handle from the given PI queue
    *ZeCmdList = QueueGroup.getImmCmdList()->first;
    *NativeHandleDesc = true;
  } else {
    auto ZeQueue = ur_cast<ze_command_queue_handle_t *>(NativeHandle);
    // Extract a Level Zero compute queue handle from the given PI queue
    uint32_t QueueGroupOrdinalUnused;
    *ZeQueue = QueueGroup.getZeQueue(&QueueGroupOrdinalUnused);
    *NativeHandleDesc = false;
  }
  return PI_SUCCESS;
}

void _pi_queue::pi_queue_group_t::setImmCmdList(
    ze_command_list_handle_t ZeCommandList) {
  ImmCmdLists = std::vector<pi_command_list_ptr_t>(
      1,
      Queue->CommandListMap
          .insert(std::pair<ze_command_list_handle_t, pi_command_list_info_t>{
              ZeCommandList, {nullptr, true, false, nullptr, 0}})
          .first);
}

pi_result piextQueueCreateWithNativeHandle(pi_native_handle NativeHandle,
                                           int32_t NativeHandleDesc,
                                           pi_context Context, pi_device Device,
                                           bool OwnNativeHandle,
                                           pi_queue_properties *Properties,
                                           pi_queue *Queue) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  // The NativeHandleDesc has value if if the native handle is an immediate
  // command list.
  if (NativeHandleDesc == 1) {
    std::vector<ze_command_queue_handle_t> ComputeQueues{nullptr};
    std::vector<ze_command_queue_handle_t> CopyQueues;

    *Queue = new _pi_queue(ComputeQueues, CopyQueues, Context, Device,
                           OwnNativeHandle, Properties[1]);
    auto &InitialGroup = (*Queue)->ComputeQueueGroupsByTID.begin()->second;
    InitialGroup.setImmCmdList(ur_cast<ze_command_list_handle_t>(NativeHandle));
  } else {
    auto ZeQueue = ur_cast<ze_command_queue_handle_t>(NativeHandle);
    // Assume this is the "0" index queue in the compute command-group.
    std::vector<ze_command_queue_handle_t> ZeQueues{ZeQueue};

    // TODO: see what we can do to correctly initialize PI queue for
    // compute vs. copy Level-Zero queue. Currently we will send
    // all commands to the "ZeQueue".
    std::vector<ze_command_queue_handle_t> ZeroCopyQueues;

    *Queue = new _pi_queue(ZeQueues, ZeroCopyQueues, Context, Device,
                           OwnNativeHandle, Properties[1]);
  }
  (*Queue)->UsingImmCmdLists = (NativeHandleDesc == 1);

  return PI_SUCCESS;
}

// If indirect access tracking is enabled then performs reference counting,
// otherwise just calls zeMemAllocDevice.
static pi_result ZeDeviceMemAllocHelper(void **ResultPtr, pi_context Context,
                                        pi_device Device, size_t Size) {
  pi_platform Plt = Device->Platform;
  std::unique_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex,
                                                 std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    // Lock the mutex which is guarding contexts container in the platform.
    // This prevents new kernels from being submitted in any context while
    // we are in the process of allocating a memory, this is needed to
    // properly capture allocations by kernels with indirect access.
    ContextsLock.lock();
    // We are going to defer memory release if there are kernels with
    // indirect access, that is why explicitly retain context to be sure
    // that it is released after all memory allocations in this context are
    // released.
    PI_CALL(piContextRetain(Context));
  }

  ze_device_mem_alloc_desc_t ZeDesc = {};
  ZeDesc.flags = 0;
  ZeDesc.ordinal = 0;
  ZE_CALL(zeMemAllocDevice,
          (Context->ZeContext, &ZeDesc, Size, 1, Device->ZeDevice, ResultPtr));

  if (IndirectAccessTrackingEnabled) {
    // Keep track of all memory allocations in the context
    Context->MemAllocs.emplace(std::piecewise_construct,
                               std::forward_as_tuple(*ResultPtr),
                               std::forward_as_tuple(Context));
  }
  return PI_SUCCESS;
}

// If indirect access tracking is enabled then performs reference counting,
// otherwise just calls zeMemAllocHost.
static pi_result ZeHostMemAllocHelper(void **ResultPtr, pi_context Context,
                                      size_t Size) {
  pi_platform Plt = Context->getPlatform();
  std::unique_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex,
                                                 std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    // Lock the mutex which is guarding contexts container in the platform.
    // This prevents new kernels from being submitted in any context while
    // we are in the process of allocating a memory, this is needed to
    // properly capture allocations by kernels with indirect access.
    ContextsLock.lock();
    // We are going to defer memory release if there are kernels with
    // indirect access, that is why explicitly retain context to be sure
    // that it is released after all memory allocations in this context are
    // released.
    PI_CALL(piContextRetain(Context));
  }

  ZeStruct<ze_host_mem_alloc_desc_t> ZeDesc;
  ZeDesc.flags = 0;
  ZE_CALL(zeMemAllocHost, (Context->ZeContext, &ZeDesc, Size, 1, ResultPtr));

  if (IndirectAccessTrackingEnabled) {
    // Keep track of all memory allocations in the context
    Context->MemAllocs.emplace(std::piecewise_construct,
                               std::forward_as_tuple(*ResultPtr),
                               std::forward_as_tuple(Context));
  }
  return PI_SUCCESS;
}

pi_result piMemBufferCreate(pi_context Context, pi_mem_flags Flags, size_t Size,
                            void *HostPtr, pi_mem *RetMem,
                            const pi_mem_properties *properties) {

  // TODO: implement support for more access modes
  if (!((Flags & PI_MEM_FLAGS_ACCESS_RW) ||
        (Flags & PI_MEM_ACCESS_READ_ONLY))) {
    die("piMemBufferCreate: Level-Zero supports read-write and read-only "
        "buffer,"
        "but not other accesses (such as write-only) yet.");
  }

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(RetMem, PI_ERROR_INVALID_VALUE);

  if (properties != nullptr) {
    die("piMemBufferCreate: no mem properties goes to Level-Zero RT yet");
  }

  if (Flags & PI_MEM_FLAGS_HOST_PTR_ALLOC) {
    // Having PI_MEM_FLAGS_HOST_PTR_ALLOC for buffer requires allocation of
    // pinned host memory, see:
    // sycl/doc/extensions/supported/sycl_ext_oneapi_use_pinned_host_memory_property.asciidoc
    // We are however missing such functionality in Level Zero, so we just
    // ignore the flag for now.
    //
  }

  // If USM Import feature is enabled and hostptr is supplied,
  // import the hostptr if not already imported into USM.
  // Data transfer rate is maximized when both source and destination
  // are USM pointers. Promotion of the host pointer to USM thus
  // optimizes data transfer performance.
  bool HostPtrImported = false;
  if (ZeUSMImport.Enabled && HostPtr != nullptr &&
      (Flags & PI_MEM_FLAGS_HOST_PTR_USE) != 0) {
    // Query memory type of the host pointer
    ze_device_handle_t ZeDeviceHandle;
    ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;
    ZE_CALL(zeMemGetAllocProperties,
            (Context->ZeContext, HostPtr, &ZeMemoryAllocationProperties,
             &ZeDeviceHandle));

    // If not shared of any type, we can import the ptr
    if (ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_UNKNOWN) {
      // Promote the host ptr to USM host memory
      ze_driver_handle_t driverHandle = Context->getPlatform()->ZeDriver;
      ZeUSMImport.doZeUSMImport(driverHandle, HostPtr, Size);
      HostPtrImported = true;
    }
  }

  pi_buffer Buffer = nullptr;
  auto HostPtrOrNull =
      (Flags & PI_MEM_FLAGS_HOST_PTR_USE) ? ur_cast<char *>(HostPtr) : nullptr;
  try {
    Buffer = new _pi_buffer(Context, Size, HostPtrOrNull, HostPtrImported);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  // Initialize the buffer with user data
  if (HostPtr) {
    if ((Flags & PI_MEM_FLAGS_HOST_PTR_USE) != 0 ||
        (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) != 0) {

      // We don't yet know which device needs this buffer, so make the first
      // device in the context be the master, and hold the initial valid
      // allocation.
      char *ZeHandleDst;
      PI_CALL(Buffer->getZeHandle(ZeHandleDst, _pi_mem::write_only,
                                  Context->Devices[0]));
      if (Buffer->OnHost) {
        // Do a host to host copy.
        // For an imported HostPtr the copy is unneeded.
        if (!HostPtrImported)
          memcpy(ZeHandleDst, HostPtr, Size);
      } else {
        // Initialize the buffer synchronously with immediate offload
        // zeCommandListAppendMemoryCopy must not be called from simultaneous
        // threads with the same command list handle, so we need exclusive lock.
        std::scoped_lock<ur_mutex> Lock(Context->ImmediateCommandListMutex);
        ZE_CALL(zeCommandListAppendMemoryCopy,
                (Context->ZeCommandListInit, ZeHandleDst, HostPtr, Size,
                 nullptr, 0, nullptr));
      }
    } else if (Flags == 0 || (Flags == PI_MEM_FLAGS_ACCESS_RW)) {
      // Nothing more to do.
    } else {
      die("piMemBufferCreate: not implemented");
    }
  }

  *RetMem = Buffer;
  return PI_SUCCESS;
}

pi_result piMemGetInfo(pi_mem Mem, pi_mem_info ParamName, size_t ParamValueSize,
                       void *ParamValue, size_t *ParamValueSizeRet) {
  PI_ASSERT(Mem, PI_ERROR_INVALID_VALUE);
  // piMemImageGetInfo must be used for images, except for shared params (like
  // Context, AccessMode, etc)
  PI_ASSERT(ParamName == PI_MEM_CONTEXT || !Mem->isImage(),
            PI_ERROR_INVALID_VALUE);

  std::shared_lock<ur_shared_mutex> Lock(Mem->Mutex);
  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  switch (ParamName) {
  case PI_MEM_CONTEXT:
    return ReturnValue(Mem->Context);
  case PI_MEM_SIZE: {
    // Get size of the allocation
    auto Buffer = ur_cast<pi_buffer>(Mem);
    return ReturnValue(size_t{Buffer->Size});
  }
  default:
    die("piMemGetInfo: Parameter is not implemented");
  }

  return {};
}

pi_result piMemRetain(pi_mem Mem) {
  PI_ASSERT(Mem, PI_ERROR_INVALID_MEM_OBJECT);

  Mem->RefCount.increment();
  return PI_SUCCESS;
}

// If indirect access tracking is not enabled then this functions just performs
// zeMemFree. If indirect access tracking is enabled then reference counting is
// performed.
static pi_result ZeMemFreeHelper(pi_context Context, void *Ptr) {
  pi_platform Plt = Context->getPlatform();
  std::unique_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex,
                                                 std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    ContextsLock.lock();
    auto It = Context->MemAllocs.find(Ptr);
    if (It == std::end(Context->MemAllocs)) {
      die("All memory allocations must be tracked!");
    }
    if (!It->second.RefCount.decrementAndTest()) {
      // Memory can't be deallocated yet.
      return PI_SUCCESS;
    }

    // Reference count is zero, it is ok to free memory.
    // We don't need to track this allocation anymore.
    Context->MemAllocs.erase(It);
  }

  ZE_CALL(zeMemFree, (Context->ZeContext, Ptr));

  if (IndirectAccessTrackingEnabled)
    PI_CALL(ContextReleaseHelper(Context));

  return PI_SUCCESS;
}

static pi_result USMFreeHelper(pi_context Context, void *Ptr,
                               bool OwnZeMemHandle = true);

pi_result piMemRelease(pi_mem Mem) {
  PI_ASSERT(Mem, PI_ERROR_INVALID_MEM_OBJECT);

  if (!Mem->RefCount.decrementAndTest())
    return PI_SUCCESS;

  if (Mem->isImage()) {
    char *ZeHandleImage;
    auto Image = static_cast<pi_image>(Mem);
    if (Image->OwnZeMemHandle) {
      PI_CALL(Mem->getZeHandle(ZeHandleImage, _pi_mem::write_only));
      auto ZeResult = ZE_CALL_NOCHECK(
          zeImageDestroy, (ur_cast<ze_image_handle_t>(ZeHandleImage)));
      // Gracefully handle the case that L0 was already unloaded.
      if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
        return mapError(ZeResult);
    }
  } else {
    auto Buffer = static_cast<pi_buffer>(Mem);
    Buffer->free();
  }
  delete Mem;

  return PI_SUCCESS;
}

static pi_result pi2zeImageDesc(const pi_image_format *ImageFormat,
                                const pi_image_desc *ImageDesc,
                                ZeStruct<ze_image_desc_t> &ZeImageDesc) {
  ze_image_format_type_t ZeImageFormatType;
  size_t ZeImageFormatTypeSize;
  switch (ImageFormat->image_channel_data_type) {
  case PI_IMAGE_CHANNEL_TYPE_FLOAT:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_FLOAT;
    ZeImageFormatTypeSize = 32;
    break;
  case PI_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_FLOAT;
    ZeImageFormatTypeSize = 16;
    break;
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
    ZeImageFormatTypeSize = 32;
    break;
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
    ZeImageFormatTypeSize = 16;
    break;
  case PI_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
    ZeImageFormatTypeSize = 8;
    break;
  case PI_IMAGE_CHANNEL_TYPE_UNORM_INT16:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UNORM;
    ZeImageFormatTypeSize = 16;
    break;
  case PI_IMAGE_CHANNEL_TYPE_UNORM_INT8:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UNORM;
    ZeImageFormatTypeSize = 8;
    break;
  case PI_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SINT;
    ZeImageFormatTypeSize = 32;
    break;
  case PI_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SINT;
    ZeImageFormatTypeSize = 16;
    break;
  case PI_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SINT;
    ZeImageFormatTypeSize = 8;
    break;
  case PI_IMAGE_CHANNEL_TYPE_SNORM_INT16:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SNORM;
    ZeImageFormatTypeSize = 16;
    break;
  case PI_IMAGE_CHANNEL_TYPE_SNORM_INT8:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SNORM;
    ZeImageFormatTypeSize = 8;
    break;
  default:
    urPrint("piMemImageCreate: unsupported image data type: data type = %d\n",
            ImageFormat->image_channel_data_type);
    return PI_ERROR_INVALID_VALUE;
  }

  // TODO: populate the layout mapping
  ze_image_format_layout_t ZeImageFormatLayout;
  switch (ImageFormat->image_channel_order) {
  case PI_IMAGE_CHANNEL_ORDER_RGBA:
    switch (ZeImageFormatTypeSize) {
    case 8:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8;
      break;
    case 16:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16;
      break;
    case 32:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32;
      break;
    default:
      urPrint("piMemImageCreate: unexpected data type Size\n");
      return PI_ERROR_INVALID_VALUE;
    }
    break;
  default:
    urPrint("format layout = %d\n", ImageFormat->image_channel_order);
    die("piMemImageCreate: unsupported image format layout\n");
    break;
  }

  ze_image_format_t ZeFormatDesc = {
      ZeImageFormatLayout, ZeImageFormatType,
      // TODO: are swizzles deducted from image_format->image_channel_order?
      ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_G,
      ZE_IMAGE_FORMAT_SWIZZLE_B, ZE_IMAGE_FORMAT_SWIZZLE_A};

  ze_image_type_t ZeImageType;
  switch (ImageDesc->image_type) {
  case PI_MEM_TYPE_IMAGE1D:
    ZeImageType = ZE_IMAGE_TYPE_1D;
    break;
  case PI_MEM_TYPE_IMAGE2D:
    ZeImageType = ZE_IMAGE_TYPE_2D;
    break;
  case PI_MEM_TYPE_IMAGE3D:
    ZeImageType = ZE_IMAGE_TYPE_3D;
    break;
  case PI_MEM_TYPE_IMAGE1D_ARRAY:
    ZeImageType = ZE_IMAGE_TYPE_1DARRAY;
    break;
  case PI_MEM_TYPE_IMAGE2D_ARRAY:
    ZeImageType = ZE_IMAGE_TYPE_2DARRAY;
    break;
  default:
    urPrint("piMemImageCreate: unsupported image type\n");
    return PI_ERROR_INVALID_VALUE;
  }

  ZeImageDesc.arraylevels = 0;
  ZeImageDesc.flags = 0;
  ZeImageDesc.type = ZeImageType;
  ZeImageDesc.format = ZeFormatDesc;
  ZeImageDesc.width = ur_cast<uint32_t>(ImageDesc->image_width);
  ZeImageDesc.height = ur_cast<uint32_t>(ImageDesc->image_height);
  ZeImageDesc.depth = ur_cast<uint32_t>(ImageDesc->image_depth);
  ZeImageDesc.arraylevels = ur_cast<uint32_t>(ImageDesc->image_array_size);
  ZeImageDesc.miplevels = ImageDesc->num_mip_levels;

  return PI_SUCCESS;
}

pi_result piMemImageCreate(pi_context Context, pi_mem_flags Flags,
                           const pi_image_format *ImageFormat,
                           const pi_image_desc *ImageDesc, void *HostPtr,
                           pi_mem *RetImage) {

  // TODO: implement read-only, write-only
  if ((Flags & PI_MEM_FLAGS_ACCESS_RW) == 0) {
    die("piMemImageCreate: Level-Zero implements only read-write buffer,"
        "no read-only or write-only yet.");
  }
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(RetImage, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(ImageFormat, PI_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);

  ZeStruct<ze_image_desc_t> ZeImageDesc;
  pi_result DescriptionResult =
      pi2zeImageDesc(ImageFormat, ImageDesc, ZeImageDesc);
  if (DescriptionResult != PI_SUCCESS)
    return DescriptionResult;

  std::shared_lock<ur_shared_mutex> Lock(Context->Mutex);

  // Currently we have the "0" device in context with mutliple root devices to
  // own the image.
  // TODO: Implement explicit copying for acessing the image from other devices
  // in the context.
  pi_device Device = Context->SingleRootDevice ? Context->SingleRootDevice
                                               : Context->Devices[0];
  ze_image_handle_t ZeHImage;
  ZE_CALL(zeImageCreate,
          (Context->ZeContext, Device->ZeDevice, &ZeImageDesc, &ZeHImage));

  try {
    auto ZePIImage = new _pi_image(Context, ZeHImage, /*OwnNativeHandle=*/true);
    *RetImage = ZePIImage;

#ifndef NDEBUG
    ZePIImage->ZeImageDesc = ZeImageDesc;
#endif // !NDEBUG

    if ((Flags & PI_MEM_FLAGS_HOST_PTR_USE) != 0 ||
        (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) != 0) {
      // Initialize image synchronously with immediate offload.
      // zeCommandListAppendImageCopyFromMemory must not be called from
      // simultaneous threads with the same command list handle, so we need
      // exclusive lock.
      std::scoped_lock<ur_mutex> Lock(Context->ImmediateCommandListMutex);
      ZE_CALL(zeCommandListAppendImageCopyFromMemory,
              (Context->ZeCommandListInit, ZeHImage, HostPtr, nullptr, nullptr,
               0, nullptr));
    }
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

pi_result piextMemGetNativeHandle(pi_mem Mem, pi_native_handle *NativeHandle) {
  PI_ASSERT(Mem, PI_ERROR_INVALID_MEM_OBJECT);
  std::shared_lock<ur_shared_mutex> Guard(Mem->Mutex);
  char *ZeHandle;
  PI_CALL(Mem->getZeHandle(ZeHandle, _pi_mem::read_write));
  *NativeHandle = ur_cast<pi_native_handle>(ZeHandle);
  return PI_SUCCESS;
}

pi_result piextMemCreateWithNativeHandle(pi_native_handle NativeHandle,
                                         pi_context Context,
                                         bool ownNativeHandle, pi_mem *Mem) {
  PI_ASSERT(Mem, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  std::shared_lock<ur_shared_mutex> Lock(Context->Mutex);

  // Get base of the allocation
  void *Base;
  size_t Size;
  void *Ptr = ur_cast<void *>(NativeHandle);
  ZE_CALL(zeMemGetAddressRange, (Context->ZeContext, Ptr, &Base, &Size));
  PI_ASSERT(Ptr == Base, PI_ERROR_INVALID_VALUE);

  ZeStruct<ze_memory_allocation_properties_t> ZeMemProps;
  ze_device_handle_t ZeDevice = nullptr;
  ZE_CALL(zeMemGetAllocProperties,
          (Context->ZeContext, Ptr, &ZeMemProps, &ZeDevice));

  // Check type of the allocation
  switch (ZeMemProps.type) {
  case ZE_MEMORY_TYPE_HOST:
  case ZE_MEMORY_TYPE_SHARED:
  case ZE_MEMORY_TYPE_DEVICE:
    break;
  case ZE_MEMORY_TYPE_UNKNOWN:
    // Memory allocation is unrelated to the context
    return PI_ERROR_INVALID_CONTEXT;
  default:
    die("Unexpected memory type");
  }

  pi_device Device = nullptr;
  if (ZeDevice) {
    Device = Context->getPlatform()->getDeviceFromNativeHandle(ZeDevice);
    PI_ASSERT(Context->isValidDevice(Device), PI_ERROR_INVALID_CONTEXT);
  }

  try {
    *Mem = new _pi_buffer(Context, Size, Device, ur_cast<char *>(NativeHandle),
                          ownNativeHandle);

    pi_platform Plt = Context->getPlatform();
    std::unique_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex,
                                                   std::defer_lock);
    // If we don't own the native handle then we can't control deallocation of
    // that memory so there is no point of keeping track of the memory
    // allocation for deferred memory release in the mode when indirect access
    // tracking is enabled.
    if (IndirectAccessTrackingEnabled && ownNativeHandle) {
      // We need to keep track of all memory allocations in the context
      ContextsLock.lock();
      // Retain context to be sure that it is released after all memory
      // allocations in this context are released.
      PI_CALL(piContextRetain(Context));

      Context->MemAllocs.emplace(
          std::piecewise_construct, std::forward_as_tuple(Ptr),
          std::forward_as_tuple(Context, ownNativeHandle));
    }
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  // Initialize the buffer as necessary
  auto Buffer = ur_cast<pi_buffer>(*Mem);
  if (Device) {
    // If this allocation is on a device, then we re-use it for the buffer.
    // Nothing to do.
  } else if (Buffer->OnHost) {
    // If this is host allocation and buffer always stays on host there
    // nothing more to do.
  } else {
    // In all other cases (shared allocation, or host allocation that cannot
    // represent the buffer in this context) copy the data to a newly
    // created device allocation.
    char *ZeHandleDst;
    PI_CALL(Buffer->getZeHandle(ZeHandleDst, _pi_mem::write_only, Device));

    // zeCommandListAppendMemoryCopy must not be called from simultaneous
    // threads with the same command list handle, so we need exclusive lock.
    std::scoped_lock<ur_mutex> Lock(Context->ImmediateCommandListMutex);
    ZE_CALL(zeCommandListAppendMemoryCopy,
            (Context->ZeCommandListInit, ZeHandleDst, Ptr, Size, nullptr, 0,
             nullptr));
  }

  return PI_SUCCESS;
}

pi_result piextMemImageCreateWithNativeHandle(
    pi_native_handle NativeHandle, pi_context Context, bool OwnNativeHandle,
    [[maybe_unused]] const pi_image_format *ImageFormat,
    [[maybe_unused]] const pi_image_desc *ImageDesc, pi_mem *RetImage) {

  PI_ASSERT(RetImage, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  std::shared_lock<ur_shared_mutex> Lock(Context->Mutex);

  ze_image_handle_t ZeHImage = ur_cast<ze_image_handle_t>(NativeHandle);

  try {
    auto ZePIImage = new _pi_image(Context, ZeHImage, OwnNativeHandle);
    *RetImage = ZePIImage;

#ifndef NDEBUG
    ZeStruct<ze_image_desc_t> ZeImageDesc;
    pi_result DescriptionResult =
        pi2zeImageDesc(ImageFormat, ImageDesc, ZeImageDesc);
    if (DescriptionResult != PI_SUCCESS)
      return DescriptionResult;

    ZePIImage->ZeImageDesc = ZeImageDesc;
#endif // !NDEBUG

  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piProgramCreate(pi_context Context, const void *ILBytes,
                          size_t Length, pi_program *Program) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(ILBytes && Length, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);

  // NOTE: the Level Zero module creation is also building the program, so we
  // are deferring it until the program is ready to be built.

  try {
    *Program = new _pi_program(_pi_program::IL, Context, ILBytes, Length);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

pi_result piProgramCreateWithBinary(
    pi_context Context, pi_uint32 NumDevices, const pi_device *DeviceList,
    const size_t *Lengths, const unsigned char **Binaries,
    size_t NumMetadataEntries, const pi_device_binary_property *Metadata,
    pi_int32 *BinaryStatus, pi_program *Program) {
  (void)Metadata;
  (void)NumMetadataEntries;

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(DeviceList && NumDevices, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Binaries && Lengths, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);

  // For now we support only one device.
  if (NumDevices != 1) {
    urPrint("piProgramCreateWithBinary: level_zero supports only one device.");
    return PI_ERROR_INVALID_VALUE;
  }
  if (!Binaries[0] || !Lengths[0]) {
    if (BinaryStatus)
      *BinaryStatus = PI_ERROR_INVALID_VALUE;
    return PI_ERROR_INVALID_VALUE;
  }

  size_t Length = Lengths[0];
  auto Binary = Binaries[0];

  // In OpenCL, clCreateProgramWithBinary() can be used to load any of the
  // following: "program executable", "compiled program", or "library of
  // compiled programs".  In addition, the loaded program can be either
  // IL (SPIR-v) or native device code.  For now, we assume that
  // piProgramCreateWithBinary() is only used to load a "program executable"
  // as native device code.
  // If we wanted to support all the same cases as OpenCL, we would need to
  // somehow examine the binary image to distinguish the cases.  Alternatively,
  // we could change the PI interface and have the caller pass additional
  // information to distinguish the cases.

  try {
    *Program = new _pi_program(_pi_program::Native, Context, Binary, Length);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  if (BinaryStatus)
    *BinaryStatus = PI_SUCCESS;
  return PI_SUCCESS;
}

pi_result piclProgramCreateWithSource(pi_context Context, pi_uint32 Count,
                                      const char **Strings,
                                      const size_t *Lengths,
                                      pi_program *RetProgram) {

  (void)Context;
  (void)Count;
  (void)Strings;
  (void)Lengths;
  (void)RetProgram;
  urPrint("piclProgramCreateWithSource: not supported in Level Zero\n");
  return PI_ERROR_INVALID_OPERATION;
}

pi_result piProgramGetInfo(pi_program Program, pi_program_info ParamName,
                           size_t ParamValueSize, void *ParamValue,
                           size_t *ParamValueSizeRet) {

  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_PROGRAM_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Program->RefCount.load()});
  case PI_PROGRAM_INFO_NUM_DEVICES:
    // TODO: return true number of devices this program exists for.
    return ReturnValue(pi_uint32{1});
  case PI_PROGRAM_INFO_DEVICES:
    // TODO: return all devices this program exists for.
    return ReturnValue(Program->Context->Devices[0]);
  case PI_PROGRAM_INFO_BINARY_SIZES: {
    std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
    size_t SzBinary;
    if (Program->State == _pi_program::IL ||
        Program->State == _pi_program::Native ||
        Program->State == _pi_program::Object) {
      SzBinary = Program->CodeLength;
    } else if (Program->State == _pi_program::Exe) {
      ZE_CALL(zeModuleGetNativeBinary, (Program->ZeModule, &SzBinary, nullptr));
    } else {
      return PI_ERROR_INVALID_PROGRAM;
    }
    // This is an array of 1 element, initialized as if it were scalar.
    return ReturnValue(size_t{SzBinary});
  }
  case PI_PROGRAM_INFO_BINARIES: {
    // The caller sets "ParamValue" to an array of pointers, one for each
    // device.  Since Level Zero supports only one device, there is only one
    // pointer.  If the pointer is NULL, we don't do anything.  Otherwise, we
    // copy the program's binary image to the buffer at that pointer.
    uint8_t **PBinary = ur_cast<uint8_t **>(ParamValue);
    if (!PBinary[0])
      break;

    std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
    if (Program->State == _pi_program::IL ||
        Program->State == _pi_program::Native ||
        Program->State == _pi_program::Object) {
      std::memcpy(PBinary[0], Program->Code.get(), Program->CodeLength);
    } else if (Program->State == _pi_program::Exe) {
      size_t SzBinary = 0;
      ZE_CALL(zeModuleGetNativeBinary,
              (Program->ZeModule, &SzBinary, PBinary[0]));
    } else {
      return PI_ERROR_INVALID_PROGRAM;
    }
    break;
  }
  case PI_PROGRAM_INFO_NUM_KERNELS: {
    std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
    uint32_t NumKernels;
    if (Program->State == _pi_program::IL ||
        Program->State == _pi_program::Native ||
        Program->State == _pi_program::Object) {
      return PI_ERROR_INVALID_PROGRAM_EXECUTABLE;
    } else if (Program->State == _pi_program::Exe) {
      NumKernels = 0;
      ZE_CALL(zeModuleGetKernelNames,
              (Program->ZeModule, &NumKernels, nullptr));
    } else {
      return PI_ERROR_INVALID_PROGRAM;
    }
    return ReturnValue(size_t{NumKernels});
  }
  case PI_PROGRAM_INFO_KERNEL_NAMES:
    try {
      std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
      std::string PINames{""};
      if (Program->State == _pi_program::IL ||
          Program->State == _pi_program::Native ||
          Program->State == _pi_program::Object) {
        return PI_ERROR_INVALID_PROGRAM_EXECUTABLE;
      } else if (Program->State == _pi_program::Exe) {
        uint32_t Count = 0;
        ZE_CALL(zeModuleGetKernelNames, (Program->ZeModule, &Count, nullptr));
        std::unique_ptr<const char *[]> PNames(new const char *[Count]);
        ZE_CALL(zeModuleGetKernelNames,
                (Program->ZeModule, &Count, PNames.get()));
        for (uint32_t I = 0; I < Count; ++I) {
          PINames += (I > 0 ? ";" : "");
          PINames += PNames[I];
        }
      } else {
        return PI_ERROR_INVALID_PROGRAM;
      }
      return ReturnValue(PINames.c_str());
    } catch (const std::bad_alloc &) {
      return PI_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
  default:
    die("piProgramGetInfo: not implemented");
  }

  return PI_SUCCESS;
}

pi_result piProgramLink(pi_context Context, pi_uint32 NumDevices,
                        const pi_device *DeviceList, const char *Options,
                        pi_uint32 NumInputPrograms,
                        const pi_program *InputPrograms,
                        void (*PFnNotify)(pi_program Program, void *UserData),
                        void *UserData, pi_program *RetProgram) {
  // We only support one device with Level Zero currently.
  if (NumDevices != 1) {
    urPrint("piProgramLink: level_zero supports only one device.");
    return PI_ERROR_INVALID_VALUE;
  }

  // We do not support any link flags at this time because the Level Zero API
  // does not have any way to pass flags that are specific to linking.
  if (Options && *Options != '\0') {
    std::string ErrorMessage(
        "Level Zero does not support kernel link flags: \"");
    ErrorMessage.append(Options);
    ErrorMessage.push_back('\"');
    pi_program Program =
        new _pi_program(_pi_program::Invalid, Context, ErrorMessage);
    *RetProgram = Program;
    return PI_ERROR_LINK_PROGRAM_FAILURE;
  }

  // Validate input parameters.
  PI_ASSERT(DeviceList, PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(Context->isValidDevice(DeviceList[0]), PI_ERROR_INVALID_DEVICE);
  PI_ASSERT(!PFnNotify && !UserData, PI_ERROR_INVALID_VALUE);
  if (NumInputPrograms == 0 || InputPrograms == nullptr)
    return PI_ERROR_INVALID_VALUE;

  pi_result PiResult = PI_SUCCESS;
  try {
    // Acquire a "shared" lock on each of the input programs, and also validate
    // that they are all in Object state.
    //
    // There is no danger of deadlock here even if two threads call
    // piProgramLink simultaneously with the same input programs in a different
    // order.  If we were acquiring these with "exclusive" access, this could
    // lead to a classic lock ordering deadlock.  However, there is no such
    // deadlock potential with "shared" access.  There could also be a deadlock
    // potential if there was some other code that holds more than one of these
    // locks simultaneously with "exclusive" access.  However, there is no such
    // code like that, so this is also not a danger.
    std::vector<std::shared_lock<ur_shared_mutex>> Guards(NumInputPrograms);
    for (pi_uint32 I = 0; I < NumInputPrograms; I++) {
      std::shared_lock<ur_shared_mutex> Guard(InputPrograms[I]->Mutex);
      Guards[I].swap(Guard);
      if (InputPrograms[I]->State != _pi_program::Object) {
        return PI_ERROR_INVALID_OPERATION;
      }
    }

    // Previous calls to piProgramCompile did not actually compile the SPIR-V.
    // Instead, we postpone compilation until this point, when all the modules
    // are linked together.  By doing compilation and linking together, the JIT
    // compiler is able see all modules and do cross-module optimizations.
    //
    // Construct a ze_module_program_exp_desc_t which contains information about
    // all of the modules that will be linked together.
    ZeStruct<ze_module_program_exp_desc_t> ZeExtModuleDesc;
    std::vector<size_t> CodeSizes(NumInputPrograms);
    std::vector<const uint8_t *> CodeBufs(NumInputPrograms);
    std::vector<const char *> BuildFlagPtrs(NumInputPrograms);
    std::vector<const ze_module_constants_t *> SpecConstPtrs(NumInputPrograms);
    std::vector<_pi_program::SpecConstantShim> SpecConstShims;
    SpecConstShims.reserve(NumInputPrograms);

    for (pi_uint32 I = 0; I < NumInputPrograms; I++) {
      pi_program Program = InputPrograms[I];
      CodeSizes[I] = Program->CodeLength;
      CodeBufs[I] = Program->Code.get();
      BuildFlagPtrs[I] = Program->BuildFlags.c_str();
      SpecConstShims.emplace_back(Program);
      SpecConstPtrs[I] = SpecConstShims[I].ze();
    }

    ZeExtModuleDesc.count = NumInputPrograms;
    ZeExtModuleDesc.inputSizes = CodeSizes.data();
    ZeExtModuleDesc.pInputModules = CodeBufs.data();
    ZeExtModuleDesc.pBuildFlags = BuildFlagPtrs.data();
    ZeExtModuleDesc.pConstants = SpecConstPtrs.data();

    ZeStruct<ze_module_desc_t> ZeModuleDesc;
    ZeModuleDesc.pNext = &ZeExtModuleDesc;
    ZeModuleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;

    // This works around a bug in the Level Zero driver.  When "ZE_DEBUG=-1",
    // the driver does validation of the API calls, and it expects
    // "pInputModule" to be non-NULL and "inputSize" to be non-zero.  This
    // validation is wrong when using the "ze_module_program_exp_desc_t"
    // extension because those fields are supposed to be ignored.  As a
    // workaround, set both fields to 1.
    //
    // TODO: Remove this workaround when the driver is fixed.
    ZeModuleDesc.pInputModule = reinterpret_cast<const uint8_t *>(1);
    ZeModuleDesc.inputSize = 1;

    // We need a Level Zero extension to compile multiple programs together into
    // a single Level Zero module.  However, we don't need that extension if
    // there happens to be only one input program.
    //
    // The "|| (NumInputPrograms == 1)" term is a workaround for a bug in the
    // Level Zero driver.  The driver's "ze_module_program_exp_desc_t"
    // extension should work even in the case when there is just one input
    // module.  However, there is currently a bug in the driver that leads to a
    // crash.  As a workaround, do not use the extension when there is one
    // input module.
    //
    // TODO: Remove this workaround when the driver is fixed.
    if (!DeviceList[0]->Platform->ZeDriverModuleProgramExtensionFound ||
        (NumInputPrograms == 1)) {
      if (NumInputPrograms == 1) {
        ZeModuleDesc.pNext = nullptr;
        ZeModuleDesc.inputSize = ZeExtModuleDesc.inputSizes[0];
        ZeModuleDesc.pInputModule = ZeExtModuleDesc.pInputModules[0];
        ZeModuleDesc.pBuildFlags = ZeExtModuleDesc.pBuildFlags[0];
        ZeModuleDesc.pConstants = ZeExtModuleDesc.pConstants[0];
      } else {
        urPrint("piProgramLink: level_zero driver does not have static linking "
                "support.");
        return PI_ERROR_INVALID_VALUE;
      }
    }

    // Call the Level Zero API to compile, link, and create the module.
    ze_device_handle_t ZeDevice = DeviceList[0]->ZeDevice;
    ze_context_handle_t ZeContext = Context->ZeContext;
    ze_module_handle_t ZeModule = nullptr;
    ze_module_build_log_handle_t ZeBuildLog = nullptr;
    ze_result_t ZeResult =
        ZE_CALL_NOCHECK(zeModuleCreate, (ZeContext, ZeDevice, &ZeModuleDesc,
                                         &ZeModule, &ZeBuildLog));

    // We still create a _pi_program object even if there is a BUILD_FAILURE
    // because we need the object to hold the ZeBuildLog.  There is no build
    // log created for other errors, so we don't create an object.
    PiResult = mapError(ZeResult);
    if (ZeResult != ZE_RESULT_SUCCESS &&
        ZeResult != ZE_RESULT_ERROR_MODULE_BUILD_FAILURE) {
      return PiResult;
    }

    // The call to zeModuleCreate does not report an error if there are
    // unresolved symbols because it thinks these could be resolved later via a
    // call to zeModuleDynamicLink.  However, modules created with piProgramLink
    // are supposed to be fully linked and ready to use.  Therefore, do an extra
    // check now for unresolved symbols.  Note that we still create a
    // _pi_program if there are unresolved symbols because the ZeBuildLog tells
    // which symbols are unresolved.
    if (ZeResult == ZE_RESULT_SUCCESS) {
      ZeResult = checkUnresolvedSymbols(ZeModule, &ZeBuildLog);
      if (ZeResult == ZE_RESULT_ERROR_MODULE_LINK_FAILURE) {
        PiResult = PI_ERROR_LINK_PROGRAM_FAILURE;
      } else if (ZeResult != ZE_RESULT_SUCCESS) {
        return mapError(ZeResult);
      }
    }

    _pi_program::state State =
        (PiResult == PI_SUCCESS) ? _pi_program::Exe : _pi_program::Invalid;
    *RetProgram = new _pi_program(State, Context, ZeModule, ZeBuildLog);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PiResult;
}

pi_result piProgramCompile(
    pi_program Program, pi_uint32 NumDevices, const pi_device *DeviceList,
    const char *Options, pi_uint32 NumInputHeaders,
    const pi_program *InputHeaders, const char **HeaderIncludeNames,
    void (*PFnNotify)(pi_program Program, void *UserData), void *UserData) {
  (void)NumInputHeaders;
  (void)InputHeaders;
  (void)HeaderIncludeNames;

  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);

  if ((NumDevices && !DeviceList) || (!NumDevices && DeviceList))
    return PI_ERROR_INVALID_VALUE;

  // These aren't supported.
  PI_ASSERT(!PFnNotify && !UserData, PI_ERROR_INVALID_VALUE);

  std::scoped_lock<ur_shared_mutex> Guard(Program->Mutex);

  // It's only valid to compile a program created from IL (we don't support
  // programs created from source code).
  //
  // The OpenCL spec says that the header parameters are ignored when compiling
  // IL programs, so we don't validate them.
  if (Program->State != _pi_program::IL)
    return PI_ERROR_INVALID_OPERATION;

  // We don't compile anything now.  Instead, we delay compilation until
  // piProgramLink, where we do both compilation and linking as a single step.
  // This produces better code because the driver can do cross-module
  // optimizations.  Therefore, we just remember the compilation flags, so we
  // can use them later.
  if (Options)
    Program->BuildFlags = Options;
  Program->State = _pi_program::Object;

  return PI_SUCCESS;
}

pi_result piProgramBuild(pi_program Program, pi_uint32 NumDevices,
                         const pi_device *DeviceList, const char *Options,
                         void (*PFnNotify)(pi_program Program, void *UserData),
                         void *UserData) {

  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);
  if ((NumDevices && !DeviceList) || (!NumDevices && DeviceList))
    return PI_ERROR_INVALID_VALUE;

  // We only support build to one device with Level Zero now.
  // TODO: we should eventually build to the possibly multiple root
  // devices in the context.
  if (NumDevices != 1) {
    urPrint("piProgramBuild: level_zero supports only one device.");
    return PI_ERROR_INVALID_VALUE;
  }

  // These aren't supported.
  PI_ASSERT(!PFnNotify && !UserData, PI_ERROR_INVALID_VALUE);

  std::scoped_lock<ur_shared_mutex> Guard(Program->Mutex);
  // Check if device belongs to associated context.
  PI_ASSERT(Program->Context, PI_ERROR_INVALID_PROGRAM);
  PI_ASSERT(Program->Context->isValidDevice(DeviceList[0]),
            PI_ERROR_INVALID_VALUE);

  // It is legal to build a program created from either IL or from native
  // device code.
  if (Program->State != _pi_program::IL &&
      Program->State != _pi_program::Native)
    return PI_ERROR_INVALID_OPERATION;

  // We should have either IL or native device code.
  PI_ASSERT(Program->Code, PI_ERROR_INVALID_PROGRAM);

  // Ask Level Zero to build and load the native code onto the device.
  ZeStruct<ze_module_desc_t> ZeModuleDesc;
  _pi_program::SpecConstantShim Shim(Program);
  ZeModuleDesc.format = (Program->State == _pi_program::IL)
                            ? ZE_MODULE_FORMAT_IL_SPIRV
                            : ZE_MODULE_FORMAT_NATIVE;
  ZeModuleDesc.inputSize = Program->CodeLength;
  ZeModuleDesc.pInputModule = Program->Code.get();
  ZeModuleDesc.pBuildFlags = Options;
  ZeModuleDesc.pConstants = Shim.ze();

  ze_device_handle_t ZeDevice = DeviceList[0]->ZeDevice;
  ze_context_handle_t ZeContext = Program->Context->ZeContext;
  ze_module_handle_t ZeModule = nullptr;

  pi_result Result = PI_SUCCESS;
  Program->State = _pi_program::Exe;
  ze_result_t ZeResult =
      ZE_CALL_NOCHECK(zeModuleCreate, (ZeContext, ZeDevice, &ZeModuleDesc,
                                       &ZeModule, &Program->ZeBuildLog));
  if (ZeResult != ZE_RESULT_SUCCESS) {
    // We adjust pi_program below to avoid attempting to release zeModule when
    // RT calls piProgramRelease().
    Program->State = _pi_program::Invalid;
    Result = mapError(ZeResult);
    if (ZeModule) {
      ZE_CALL_NOCHECK(zeModuleDestroy, (ZeModule));
      ZeModule = nullptr;
    }
  } else {
    // The call to zeModuleCreate does not report an error if there are
    // unresolved symbols because it thinks these could be resolved later via a
    // call to zeModuleDynamicLink.  However, modules created with
    // piProgramBuild are supposed to be fully linked and ready to use.
    // Therefore, do an extra check now for unresolved symbols.
    ZeResult = checkUnresolvedSymbols(ZeModule, &Program->ZeBuildLog);
    if (ZeResult != ZE_RESULT_SUCCESS) {
      Program->State = _pi_program::Invalid;
      Result = (ZeResult == ZE_RESULT_ERROR_MODULE_LINK_FAILURE)
                   ? PI_ERROR_BUILD_PROGRAM_FAILURE
                   : mapError(ZeResult);
      if (ZeModule) {
        ZE_CALL_NOCHECK(zeModuleDestroy, (ZeModule));
        ZeModule = nullptr;
      }
    }
  }

  // We no longer need the IL / native code.
  Program->Code.reset();
  Program->ZeModule = ZeModule;
  return Result;
}

pi_result piProgramGetBuildInfo(pi_program Program, pi_device Device,
                                pi_program_build_info ParamName,
                                size_t ParamValueSize, void *ParamValue,
                                size_t *ParamValueSizeRet) {
  (void)Device;

  std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  if (ParamName == PI_PROGRAM_BUILD_INFO_BINARY_TYPE) {
    pi_program_binary_type Type = PI_PROGRAM_BINARY_TYPE_NONE;
    if (Program->State == _pi_program::Object) {
      Type = PI_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;
    } else if (Program->State == _pi_program::Exe) {
      Type = PI_PROGRAM_BINARY_TYPE_EXECUTABLE;
    }
    return ReturnValue(pi_program_binary_type{Type});
  }
  if (ParamName == PI_PROGRAM_BUILD_INFO_OPTIONS) {
    // TODO: how to get module build options out of Level Zero?
    // For the programs that we compiled we can remember the options
    // passed with piProgramCompile/piProgramBuild, but what can we
    // return for programs that were built outside and registered
    // with piProgramRegister?
    return ReturnValue("");
  } else if (ParamName == PI_PROGRAM_BUILD_INFO_LOG) {
    // Check first to see if the plugin code recorded an error message.
    if (!Program->ErrorMessage.empty()) {
      return ReturnValue(Program->ErrorMessage.c_str());
    }

    // Next check if there is a Level Zero build log.
    if (Program->ZeBuildLog) {
      size_t LogSize = ParamValueSize;
      ZE_CALL(zeModuleBuildLogGetString,
              (Program->ZeBuildLog, &LogSize, ur_cast<char *>(ParamValue)));
      if (ParamValueSizeRet) {
        *ParamValueSizeRet = LogSize;
      }
      if (ParamValue) {
        // When the program build fails in piProgramBuild(), we delayed cleaning
        // up the build log because RT later calls this routine to get the
        // failed build log.
        // To avoid memory leaks, we should clean up the failed build log here
        // because RT does not create sycl::program when piProgramBuild() fails,
        // thus it won't call piProgramRelease() to clean up the build log.
        if (Program->State == _pi_program::Invalid) {
          ZE_CALL_NOCHECK(zeModuleBuildLogDestroy, (Program->ZeBuildLog));
          Program->ZeBuildLog = nullptr;
        }
      }
      return PI_SUCCESS;
    }

    // Otherwise, there is no error.  The OpenCL spec says to return an empty
    // string if there ws no previous attempt to compile, build, or link the
    // program.
    return ReturnValue("");
  } else {
    urPrint("piProgramGetBuildInfo: unsupported ParamName\n");
    return PI_ERROR_INVALID_VALUE;
  }
  return PI_SUCCESS;
}

pi_result piProgramRetain(pi_program Program) {
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);
  Program->RefCount.increment();
  return PI_SUCCESS;
}

pi_result piProgramRelease(pi_program Program) {
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);

  if (!Program->RefCount.decrementAndTest())
    return PI_SUCCESS;

  delete Program;

  return PI_SUCCESS;
}

pi_result piextProgramGetNativeHandle(pi_program Program,
                                      pi_native_handle *NativeHandle) {
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  auto ZeModule = ur_cast<ze_module_handle_t *>(NativeHandle);

  std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
  switch (Program->State) {
  case _pi_program::Exe: {
    *ZeModule = Program->ZeModule;
    break;
  }

  default:
    return PI_ERROR_INVALID_OPERATION;
  }

  return PI_SUCCESS;
}

pi_result piextProgramCreateWithNativeHandle(pi_native_handle NativeHandle,
                                             pi_context Context,
                                             bool ownNativeHandle,
                                             pi_program *Program) {
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  auto ZeModule = ur_cast<ze_module_handle_t>(NativeHandle);

  // We assume here that programs created from a native handle always
  // represent a fully linked executable (state Exe) and not an unlinked
  // executable (state Object).

  try {
    *Program =
        new _pi_program(_pi_program::Exe, Context, ZeModule, ownNativeHandle);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

_pi_program::~_pi_program() {
  // According to Level Zero Specification, all kernels and build logs
  // must be destroyed before the Module can be destroyed.  So, be sure
  // to destroy build log before destroying the module.
  if (ZeBuildLog) {
    ZE_CALL_NOCHECK(zeModuleBuildLogDestroy, (ZeBuildLog));
  }

  if (ZeModule && OwnZeModule) {
    ZE_CALL_NOCHECK(zeModuleDestroy, (ZeModule));
  }
}

// Check to see if a Level Zero module has any unresolved symbols.
//
// @param ZeModule    The module handle to check.
// @param ZeBuildLog  If there are unresolved symbols, this build log handle is
//                     modified to receive information telling which symbols
//                     are unresolved.
//
// @return ZE_RESULT_ERROR_MODULE_LINK_FAILURE indicates there are unresolved
//  symbols.  ZE_RESULT_SUCCESS indicates all symbols are resolved.  Any other
//  value indicates there was an error and we cannot tell if symbols are
//  resolved.
static ze_result_t
checkUnresolvedSymbols(ze_module_handle_t ZeModule,
                       ze_module_build_log_handle_t *ZeBuildLog) {

  // First check to see if the module has any imported symbols.  If there are
  // no imported symbols, it's not possible to have any unresolved symbols.  We
  // do this check first because we assume it's faster than the call to
  // zeModuleDynamicLink below.
  ZeStruct<ze_module_properties_t> ZeModuleProps;
  ze_result_t ZeResult =
      ZE_CALL_NOCHECK(zeModuleGetProperties, (ZeModule, &ZeModuleProps));
  if (ZeResult != ZE_RESULT_SUCCESS)
    return ZeResult;

  // If there are imported symbols, attempt to "link" the module with itself.
  // As a side effect, this will return the error
  // ZE_RESULT_ERROR_MODULE_LINK_FAILURE if there are any unresolved symbols.
  if (ZeModuleProps.flags & ZE_MODULE_PROPERTY_FLAG_IMPORTS) {
    return ZE_CALL_NOCHECK(zeModuleDynamicLink, (1, &ZeModule, ZeBuildLog));
  }
  return ZE_RESULT_SUCCESS;
}

pi_result piKernelCreate(pi_program Program, const char *KernelName,
                         pi_kernel *RetKernel) {

  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);
  PI_ASSERT(RetKernel, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(KernelName, PI_ERROR_INVALID_VALUE);

  std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
  if (Program->State != _pi_program::Exe) {
    return PI_ERROR_INVALID_PROGRAM_EXECUTABLE;
  }

  ZeStruct<ze_kernel_desc_t> ZeKernelDesc;
  ZeKernelDesc.flags = 0;
  ZeKernelDesc.pKernelName = KernelName;

  ze_kernel_handle_t ZeKernel;
  ZE_CALL(zeKernelCreate, (Program->ZeModule, &ZeKernelDesc, &ZeKernel));

  try {
    *RetKernel = new _pi_kernel(ZeKernel, true, Program);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  PI_CALL((*RetKernel)->initialize());
  return PI_SUCCESS;
}

pi_result _pi_kernel::initialize() {
  // Retain the program and context to show it's used by this kernel.
  PI_CALL(piProgramRetain(Program));
  if (IndirectAccessTrackingEnabled)
    // TODO: do piContextRetain without the guard
    PI_CALL(piContextRetain(Program->Context));

  // Set up how to obtain kernel properties when needed.
  ZeKernelProperties.Compute = [this](ze_kernel_properties_t &Properties) {
    ZE_CALL_NOCHECK(zeKernelGetProperties, (ZeKernel, &Properties));
  };

  // Cache kernel name.
  ZeKernelName.Compute = [this](std::string &Name) {
    size_t Size = 0;
    ZE_CALL_NOCHECK(zeKernelGetName, (ZeKernel, &Size, nullptr));
    char *KernelName = new char[Size];
    ZE_CALL_NOCHECK(zeKernelGetName, (ZeKernel, &Size, KernelName));
    Name = KernelName;
    delete[] KernelName;
  };

  return PI_SUCCESS;
}

pi_result piKernelSetArg(pi_kernel Kernel, pi_uint32 ArgIndex, size_t ArgSize,
                         const void *ArgValue) {

  // OpenCL: "the arg_value pointer can be NULL or point to a NULL value
  // in which case a NULL value will be used as the value for the argument
  // declared as a pointer to global or constant memory in the kernel"
  //
  // We don't know the type of the argument but it seems that the only time
  // SYCL RT would send a pointer to NULL in 'arg_value' is when the argument
  // is a NULL pointer. Treat a pointer to NULL in 'arg_value' as a NULL.
  if (ArgSize == sizeof(void *) && ArgValue &&
      *(void **)(const_cast<void *>(ArgValue)) == nullptr) {
    ArgValue = nullptr;
  }

  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  std::scoped_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  ZE_CALL(zeKernelSetArgumentValue,
          (ur_cast<ze_kernel_handle_t>(Kernel->ZeKernel),
           ur_cast<uint32_t>(ArgIndex), ur_cast<size_t>(ArgSize),
           ur_cast<const void *>(ArgValue)));

  return PI_SUCCESS;
}

// Special version of piKernelSetArg to accept pi_mem.
pi_result piextKernelSetArgMemObj(pi_kernel Kernel, pi_uint32 ArgIndex,
                                  const pi_mem *ArgValue) {
  // TODO: the better way would probably be to add a new PI API for
  // extracting native PI object from PI handle, and have SYCL
  // RT pass that directly to the regular piKernelSetArg (and
  // then remove this piextKernelSetArgMemObj).

  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  // We don't yet know the device where this kernel will next be run on.
  // Thus we can't know the actual memory allocation that needs to be used.
  // Remember the memory object being used as an argument for this kernel
  // to process it later when the device is known (at the kernel enqueue).
  //
  // TODO: for now we have to conservatively assume the access as read-write.
  //       Improve that by passing SYCL buffer accessor type into
  //       piextKernelSetArgMemObj.
  //
  std::scoped_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  // The ArgValue may be a NULL pointer in which case a NULL value is used for
  // the kernel argument declared as a pointer to global or constant memory.
  auto Arg = ArgValue ? *ArgValue : nullptr;
  Kernel->PendingArguments.push_back(
      {ArgIndex, sizeof(void *), Arg, _pi_mem::read_write});

  return PI_SUCCESS;
}

// Special version of piKernelSetArg to accept pi_sampler.
pi_result piextKernelSetArgSampler(pi_kernel Kernel, pi_uint32 ArgIndex,
                                   const pi_sampler *ArgValue) {
  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  std::scoped_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  ZE_CALL(zeKernelSetArgumentValue,
          (ur_cast<ze_kernel_handle_t>(Kernel->ZeKernel),
           ur_cast<uint32_t>(ArgIndex), sizeof(void *),
           &(*ArgValue)->ZeSampler));

  return PI_SUCCESS;
}

pi_result piKernelGetInfo(pi_kernel Kernel, pi_kernel_info ParamName,
                          size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet) {
  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  std::shared_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  switch (ParamName) {
  case PI_KERNEL_INFO_CONTEXT:
    return ReturnValue(pi_context{Kernel->Program->Context});
  case PI_KERNEL_INFO_PROGRAM:
    return ReturnValue(pi_program{Kernel->Program});
  case PI_KERNEL_INFO_FUNCTION_NAME:
    try {
      std::string &KernelName = *Kernel->ZeKernelName.operator->();
      return ReturnValue(static_cast<const char *>(KernelName.c_str()));
    } catch (const std::bad_alloc &) {
      return PI_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
  case PI_KERNEL_INFO_NUM_ARGS:
    return ReturnValue(pi_uint32{Kernel->ZeKernelProperties->numKernelArgs});
  case PI_KERNEL_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Kernel->RefCount.load()});
  case PI_KERNEL_INFO_ATTRIBUTES:
    try {
      uint32_t Size;
      ZE_CALL(zeKernelGetSourceAttributes, (Kernel->ZeKernel, &Size, nullptr));
      char *attributes = new char[Size];
      ZE_CALL(zeKernelGetSourceAttributes,
              (Kernel->ZeKernel, &Size, &attributes));
      auto Res = ReturnValue(attributes);
      delete[] attributes;
      return Res;
    } catch (const std::bad_alloc &) {
      return PI_ERROR_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
  default:
    urPrint("Unsupported ParamName in piKernelGetInfo: ParamName=%d(0x%x)\n",
            ParamName, ParamName);
    return PI_ERROR_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

pi_result piKernelGetGroupInfo(pi_kernel Kernel, pi_device Device,
                               pi_kernel_group_info ParamName,
                               size_t ParamValueSize, void *ParamValue,
                               size_t *ParamValueSizeRet) {
  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  std::shared_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  switch (ParamName) {
  case PI_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE: {
    struct {
      size_t Arr[3];
    } GlobalWorkSize = {{(Device->ZeDeviceComputeProperties->maxGroupSizeX *
                          Device->ZeDeviceComputeProperties->maxGroupCountX),
                         (Device->ZeDeviceComputeProperties->maxGroupSizeY *
                          Device->ZeDeviceComputeProperties->maxGroupCountY),
                         (Device->ZeDeviceComputeProperties->maxGroupSizeZ *
                          Device->ZeDeviceComputeProperties->maxGroupCountZ)}};
    return ReturnValue(GlobalWorkSize);
  }
  case PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
    // As of right now, L0 is missing API to query kernel and device specific
    // max work group size.
    return ReturnValue(
        pi_uint64{Device->ZeDeviceComputeProperties->maxTotalGroupSize});
  }
  case PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE: {
    struct {
      size_t Arr[3];
    } WgSize = {{Kernel->ZeKernelProperties->requiredGroupSizeX,
                 Kernel->ZeKernelProperties->requiredGroupSizeY,
                 Kernel->ZeKernelProperties->requiredGroupSizeZ}};
    return ReturnValue(WgSize);
  }
  case PI_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE:
    return ReturnValue(pi_uint32{Kernel->ZeKernelProperties->localMemSize});
  case PI_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
    return ReturnValue(size_t{Device->ZeDeviceProperties->physicalEUSimdWidth});
  }
  case PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE:
    return ReturnValue(pi_uint32{Kernel->ZeKernelProperties->privateMemSize});
  case PI_KERNEL_GROUP_INFO_NUM_REGS: {
    die("PI_KERNEL_GROUP_INFO_NUM_REGS in piKernelGetGroupInfo not "
        "implemented\n");
    break;
  }
  default:
    urPrint("Unknown ParamName in piKernelGetGroupInfo: ParamName=%d(0x%x)\n",
            ParamName, ParamName);
    return PI_ERROR_INVALID_VALUE;
  }
  return PI_SUCCESS;
}

pi_result piKernelGetSubGroupInfo(pi_kernel Kernel, pi_device Device,
                                  pi_kernel_sub_group_info ParamName,
                                  size_t InputValueSize, const void *InputValue,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {
  (void)Device;
  (void)InputValueSize;
  (void)InputValue;

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  std::shared_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  if (ParamName == PI_KERNEL_MAX_SUB_GROUP_SIZE) {
    ReturnValue(uint32_t{Kernel->ZeKernelProperties->maxSubgroupSize});
  } else if (ParamName == PI_KERNEL_MAX_NUM_SUB_GROUPS) {
    ReturnValue(uint32_t{Kernel->ZeKernelProperties->maxNumSubgroups});
  } else if (ParamName == PI_KERNEL_COMPILE_NUM_SUB_GROUPS) {
    ReturnValue(uint32_t{Kernel->ZeKernelProperties->requiredNumSubGroups});
  } else if (ParamName == PI_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL) {
    ReturnValue(uint32_t{Kernel->ZeKernelProperties->requiredSubgroupSize});
  } else {
    die("piKernelGetSubGroupInfo: parameter not implemented");
    return {};
  }
  return PI_SUCCESS;
}

pi_result piKernelRetain(pi_kernel Kernel) {

  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  Kernel->RefCount.increment();
  return PI_SUCCESS;
}

pi_result piKernelRelease(pi_kernel Kernel) {
  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  if (!Kernel->RefCount.decrementAndTest())
    return PI_SUCCESS;

  auto KernelProgram = Kernel->Program;
  if (Kernel->OwnZeKernel) {
    auto ZeResult = ZE_CALL_NOCHECK(zeKernelDestroy, (Kernel->ZeKernel));
    // Gracefully handle the case that L0 was already unloaded.
    if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
      return mapError(ZeResult);
  }
  if (IndirectAccessTrackingEnabled) {
    PI_CALL(piContextRelease(KernelProgram->Context));
  }
  // do a release on the program this kernel was part of
  PI_CALL(piProgramRelease(KernelProgram));
  delete Kernel;

  return PI_SUCCESS;
}

pi_result
piEnqueueKernelLaunch(pi_queue Queue, pi_kernel Kernel, pi_uint32 WorkDim,
                      const size_t *GlobalWorkOffset,
                      const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
                      pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *OutEvent) {
  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);
  PI_ASSERT((WorkDim > 0) && (WorkDim < 4), PI_ERROR_INVALID_WORK_DIMENSION);

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex, ur_shared_mutex> Lock(
      Queue->Mutex, Kernel->Mutex, Kernel->Program->Mutex);
  if (GlobalWorkOffset != NULL) {
    if (!Queue->Device->Platform->ZeDriverGlobalOffsetExtensionFound) {
      urPrint("No global offset extension found on this driver\n");
      return PI_ERROR_INVALID_VALUE;
    }

    ZE_CALL(zeKernelSetGlobalOffsetExp,
            (Kernel->ZeKernel, GlobalWorkOffset[0], GlobalWorkOffset[1],
             GlobalWorkOffset[2]));
  }

  // If there are any pending arguments set them now.
  for (auto &Arg : Kernel->PendingArguments) {
    // The ArgValue may be a NULL pointer in which case a NULL value is used for
    // the kernel argument declared as a pointer to global or constant memory.
    char **ZeHandlePtr = nullptr;
    if (Arg.Value) {
      PI_CALL(Arg.Value->getZeHandlePtr(ZeHandlePtr, Arg.AccessMode,
                                        Queue->Device));
    }
    ZE_CALL(zeKernelSetArgumentValue,
            (Kernel->ZeKernel, Arg.Index, Arg.Size, ZeHandlePtr));
  }
  Kernel->PendingArguments.clear();

  ze_group_count_t ZeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3];

  // global_work_size of unused dimensions must be set to 1
  PI_ASSERT(WorkDim == 3 || GlobalWorkSize[2] == 1, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(WorkDim >= 2 || GlobalWorkSize[1] == 1, PI_ERROR_INVALID_VALUE);

  if (LocalWorkSize) {
    WG[0] = ur_cast<uint32_t>(LocalWorkSize[0]);
    WG[1] = ur_cast<uint32_t>(LocalWorkSize[1]);
    WG[2] = ur_cast<uint32_t>(LocalWorkSize[2]);
  } else {
    // We can't call to zeKernelSuggestGroupSize if 64-bit GlobalWorkSize
    // values do not fit to 32-bit that the API only supports currently.
    bool SuggestGroupSize = true;
    for (int I : {0, 1, 2}) {
      if (GlobalWorkSize[I] > UINT32_MAX) {
        SuggestGroupSize = false;
      }
    }
    if (SuggestGroupSize) {
      ZE_CALL(zeKernelSuggestGroupSize,
              (Kernel->ZeKernel, GlobalWorkSize[0], GlobalWorkSize[1],
               GlobalWorkSize[2], &WG[0], &WG[1], &WG[2]));
    } else {
      for (int I : {0, 1, 2}) {
        // Try to find a I-dimension WG size that the GlobalWorkSize[I] is
        // fully divisable with. Start with the max possible size in
        // each dimension.
        uint32_t GroupSize[] = {
            Queue->Device->ZeDeviceComputeProperties->maxGroupSizeX,
            Queue->Device->ZeDeviceComputeProperties->maxGroupSizeY,
            Queue->Device->ZeDeviceComputeProperties->maxGroupSizeZ};
        GroupSize[I] = std::min(size_t(GroupSize[I]), GlobalWorkSize[I]);
        while (GlobalWorkSize[I] % GroupSize[I]) {
          --GroupSize[I];
        }
        if (GlobalWorkSize[I] / GroupSize[I] > UINT32_MAX) {
          urPrint("piEnqueueKernelLaunch: can't find a WG size "
                  "suitable for global work size > UINT32_MAX\n");
          return PI_ERROR_INVALID_WORK_GROUP_SIZE;
        }
        WG[I] = GroupSize[I];
      }
      urPrint("piEnqueueKernelLaunch: using computed WG size = {%d, %d, %d}\n",
              WG[0], WG[1], WG[2]);
    }
  }

  // TODO: assert if sizes do not fit into 32-bit?
  switch (WorkDim) {
  case 3:
    ZeThreadGroupDimensions.groupCountX =
        ur_cast<uint32_t>(GlobalWorkSize[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        ur_cast<uint32_t>(GlobalWorkSize[1] / WG[1]);
    ZeThreadGroupDimensions.groupCountZ =
        ur_cast<uint32_t>(GlobalWorkSize[2] / WG[2]);
    break;
  case 2:
    ZeThreadGroupDimensions.groupCountX =
        ur_cast<uint32_t>(GlobalWorkSize[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        ur_cast<uint32_t>(GlobalWorkSize[1] / WG[1]);
    WG[2] = 1;
    break;
  case 1:
    ZeThreadGroupDimensions.groupCountX =
        ur_cast<uint32_t>(GlobalWorkSize[0] / WG[0]);
    WG[1] = WG[2] = 1;
    break;

  default:
    urPrint("piEnqueueKernelLaunch: unsupported work_dim\n");
    return PI_ERROR_INVALID_VALUE;
  }

  // Error handling for non-uniform group size case
  if (GlobalWorkSize[0] !=
      size_t(ZeThreadGroupDimensions.groupCountX) * WG[0]) {
    urPrint("piEnqueueKernelLaunch: invalid work_dim. The range is not a "
            "multiple of the group size in the 1st dimension\n");
    return PI_ERROR_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize[1] !=
      size_t(ZeThreadGroupDimensions.groupCountY) * WG[1]) {
    urPrint("piEnqueueKernelLaunch: invalid work_dim. The range is not a "
            "multiple of the group size in the 2nd dimension\n");
    return PI_ERROR_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize[2] !=
      size_t(ZeThreadGroupDimensions.groupCountZ) * WG[2]) {
    urPrint("piEnqueueKernelLaunch: invalid work_dim. The range is not a "
            "multiple of the group size in the 3rd dimension\n");
    return PI_ERROR_INVALID_WORK_GROUP_SIZE;
  }

  ZE_CALL(zeKernelSetGroupSize, (Kernel->ZeKernel, WG[0], WG[1], WG[2]));

  bool UseCopyEngine = false;
  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
          NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine))
    return Res;

  // Get a new command list to be used on this call
  pi_command_list_ptr_t CommandList{};
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, CommandList, UseCopyEngine, true /* AllowBatching */))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  pi_event InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  pi_event *Event = OutEvent ? OutEvent : &InternalEvent;
  pi_result Res = createEventAndAssociateQueue(
      Queue, Event, PI_COMMAND_TYPE_NDRANGE_KERNEL, CommandList, IsInternal);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  // Save the kernel in the event, so that when the event is signalled
  // the code can do a piKernelRelease on this kernel.
  (*Event)->CommandData = (void *)Kernel;

  // Increment the reference count of the Kernel and indicate that the Kernel is
  // in use. Once the event has been signalled, the code in
  // CleanupCompletedEvent(Event) will do a piReleaseKernel to update the
  // reference count on the kernel, using the kernel saved in CommandData.
  PI_CALL(piKernelRetain(Kernel));

  // Add to list of kernels to be submitted
  if (IndirectAccessTrackingEnabled)
    Queue->KernelsToBeSubmitted.push_back(Kernel);

  if (Queue->UsingImmCmdLists && IndirectAccessTrackingEnabled) {
    // If using immediate commandlists then gathering of indirect
    // references and appending to the queue (which means submission)
    // must be done together.
    std::unique_lock<ur_shared_mutex> ContextsLock(
        Queue->Device->Platform->ContextsMutex, std::defer_lock);
    // We are going to submit kernels for execution. If indirect access flag is
    // set for a kernel then we need to make a snapshot of existing memory
    // allocations in all contexts in the platform. We need to lock the mutex
    // guarding the list of contexts in the platform to prevent creation of new
    // memory alocations in any context before we submit the kernel for
    // execution.
    ContextsLock.lock();
    Queue->CaptureIndirectAccesses();
    // Add the command to the command list, which implies submission.
    ZE_CALL(zeCommandListAppendLaunchKernel,
            (CommandList->first, Kernel->ZeKernel, &ZeThreadGroupDimensions,
             ZeEvent, (*Event)->WaitList.Length,
             (*Event)->WaitList.ZeEventList));
  } else {
    // Add the command to the command list for later submission.
    // No lock is needed here, unlike the immediate commandlist case above,
    // because the kernels are not actually submitted yet. Kernels will be
    // submitted only when the comamndlist is closed. Then, a lock is held.
    ZE_CALL(zeCommandListAppendLaunchKernel,
            (CommandList->first, Kernel->ZeKernel, &ZeThreadGroupDimensions,
             ZeEvent, (*Event)->WaitList.Length,
             (*Event)->WaitList.ZeEventList));
  }

  urPrint("calling zeCommandListAppendLaunchKernel() with"
          "  ZeEvent %#llx\n",
          ur_cast<std::uintptr_t>(ZeEvent));
  printZeEventList((*Event)->WaitList);

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  if (auto Res = Queue->executeCommandList(CommandList, false, true))
    return Res;

  return PI_SUCCESS;
}

pi_result piextKernelCreateWithNativeHandle(pi_native_handle NativeHandle,
                                            pi_context Context,
                                            pi_program Program,
                                            bool OwnNativeHandle,
                                            pi_kernel *Kernel) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);

  auto ZeKernel = ur_cast<ze_kernel_handle_t>(NativeHandle);
  *Kernel = new _pi_kernel(ZeKernel, OwnNativeHandle, Program);
  PI_CALL((*Kernel)->initialize());
  return PI_SUCCESS;
}

pi_result piextKernelGetNativeHandle(pi_kernel Kernel,
                                     pi_native_handle *NativeHandle) {
  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  std::shared_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  auto *ZeKernel = ur_cast<ze_kernel_handle_t *>(NativeHandle);
  *ZeKernel = Kernel->ZeKernel;
  return PI_SUCCESS;
}

//
// Events
//
pi_result
_pi_event::getOrCreateHostVisibleEvent(ze_event_handle_t &ZeHostVisibleEvent) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_EVENT);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          this->Mutex);

  if (!HostVisibleEvent) {
    if (Queue->Device->ZeEventsScope != OnDemandHostVisibleProxy)
      die("getOrCreateHostVisibleEvent: missing host-visible event");

    // Submit the command(s) signalling the proxy event to the queue.
    // We have to first submit a wait for the device-only event for which this
    // proxy is created.
    //
    // Get a new command list to be used on this call

    // We want to batch these commands to avoid extra submissions (costly)
    bool OkToBatch = true;

    pi_command_list_ptr_t CommandList{};
    if (auto Res = Queue->Context->getAvailableCommandList(
            Queue, CommandList, false /* UseCopyEngine */, OkToBatch))
      return Res;

    // Create a "proxy" host-visible event.
    auto Res = createEventAndAssociateQueue(
        Queue, &HostVisibleEvent, PI_COMMAND_TYPE_USER, CommandList,
        /* IsInternal */ false, /* HostVisible */ true);
    if (Res != PI_SUCCESS)
      return Res;

    ZE_CALL(zeCommandListAppendWaitOnEvents, (CommandList->first, 1, &ZeEvent));
    ZE_CALL(zeCommandListAppendSignalEvent,
            (CommandList->first, HostVisibleEvent->ZeEvent));

    if (auto Res = Queue->executeCommandList(CommandList, false, OkToBatch))
      return Res;
  }

  ZeHostVisibleEvent = HostVisibleEvent->ZeEvent;
  return PI_SUCCESS;
}

pi_result _pi_event::reset() {
  Queue = nullptr;
  CleanedUp = false;
  Completed = false;
  CommandData = nullptr;
  CommandType = PI_COMMAND_TYPE_USER;
  WaitList = {};
  RefCountExternal = 0;
  RefCount.reset();
  CommandList = std::nullopt;

  if (!isHostVisible())
    HostVisibleEvent = nullptr;

  ZE_CALL(zeEventHostReset, (ZeEvent));
  return PI_SUCCESS;
}

pi_event _pi_context::getEventFromContextCache(bool HostVisible,
                                               bool WithProfiling) {
  std::scoped_lock<ur_mutex> Lock(EventCacheMutex);
  auto Cache = getEventCache(HostVisible, WithProfiling);
  if (Cache->empty())
    return nullptr;

  auto It = Cache->begin();
  pi_event Event = *It;
  Cache->erase(It);
  // We have to reset event before using it.
  Event->reset();
  return Event;
}

void _pi_context::addEventToContextCache(pi_event Event) {
  std::scoped_lock<ur_mutex> Lock(EventCacheMutex);
  auto Cache =
      getEventCache(Event->isHostVisible(), Event->isProfilingEnabled());
  Cache->emplace_back(Event);
}

// Helper function for creating a PI event.
// The "Queue" argument specifies the PI queue where a command is submitted.
// The "HostVisible" argument specifies if event needs to be allocated from
// a host-visible pool.
//
static pi_result EventCreate(pi_context Context, pi_queue Queue,
                             bool HostVisible, pi_event *RetEvent) {
  bool ProfilingEnabled =
      !Queue || (Queue->Properties & PI_QUEUE_FLAG_PROFILING_ENABLE) != 0;

  if (auto CachedEvent =
          Context->getEventFromContextCache(HostVisible, ProfilingEnabled)) {
    *RetEvent = CachedEvent;
    return PI_SUCCESS;
  }

  ze_event_handle_t ZeEvent;
  ze_event_pool_handle_t ZeEventPool = {};

  size_t Index = 0;

  if (auto Res = Context->getFreeSlotInExistingOrNewPool(
          ZeEventPool, Index, HostVisible, ProfilingEnabled))
    return Res;

  ZeStruct<ze_event_desc_t> ZeEventDesc;
  ZeEventDesc.index = Index;
  ZeEventDesc.wait = 0;

  if (HostVisible) {
    ZeEventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  } else {
    //
    // Set the scope to "device" for every event. This is sufficient for
    // global device access and peer device access. If needed to be seen on
    // the host we are doing special handling, see EventsScope options.
    //
    // TODO: see if "sub-device" (ZE_EVENT_SCOPE_FLAG_SUBDEVICE) can better be
    //       used in some circumstances.
    //
    ZeEventDesc.signal = 0;
  }

  ZE_CALL(zeEventCreate, (ZeEventPool, &ZeEventDesc, &ZeEvent));

  try {
    PI_ASSERT(RetEvent, PI_ERROR_INVALID_VALUE);

    *RetEvent = new _pi_event(ZeEvent, ZeEventPool, Context,
                              PI_COMMAND_TYPE_USER, true);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  if (HostVisible)
    (*RetEvent)->HostVisibleEvent = *RetEvent;

  return PI_SUCCESS;
}

// External PI API entry
pi_result piEventCreate(pi_context Context, pi_event *RetEvent) {
  pi_result Result = EventCreate(Context, nullptr, true, RetEvent);
  (*RetEvent)->RefCountExternal++;
  if (Result != PI_SUCCESS)
    return Result;
  ZE_CALL(zeEventHostSignal, ((*RetEvent)->ZeEvent));
  return PI_SUCCESS;
}

pi_result piEventGetInfo(pi_event Event, pi_event_info ParamName,
                         size_t ParamValueSize, void *ParamValue,
                         size_t *ParamValueSizeRet) {

  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_EVENT_INFO_COMMAND_QUEUE: {
    std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
    return ReturnValue(pi_queue{Event->Queue});
  }
  case PI_EVENT_INFO_CONTEXT: {
    std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
    return ReturnValue(pi_context{Event->Context});
  }
  case PI_EVENT_INFO_COMMAND_TYPE: {
    std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
    return ReturnValue(ur_cast<pi_uint64>(Event->CommandType));
  }
  case PI_EVENT_INFO_COMMAND_EXECUTION_STATUS: {
    // Check to see if the event's Queue has an open command list due to
    // batching. If so, go ahead and close and submit it, because it is
    // possible that this is trying to query some event's status that
    // is part of the batch.  This isn't strictly required, but it seems
    // like a reasonable thing to do.
    auto Queue = Event->Queue;
    if (Queue) {
      // Lock automatically releases when this goes out of scope.
      std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);
      const auto &OpenCommandList = Queue->eventOpenCommandList(Event);
      if (OpenCommandList != Queue->CommandListMap.end()) {
        if (auto Res = Queue->executeOpenCommandList(
                OpenCommandList->second.isCopy(Queue)))
          return Res;
      }
    }

    // Level Zero has a much more explicit notion of command submission than
    // OpenCL. It doesn't happen unless the user submits a command list. We've
    // done it just above so the status is at least PI_EVENT_SUBMITTED.
    //
    // NOTE: We currently cannot tell if command is currently running, so
    // it will always show up "submitted" before it is finally "completed".
    //
    pi_int32 Result = PI_EVENT_SUBMITTED;

    // Make sure that we query a host-visible event only.
    // If one wasn't yet created then don't create it here as well, and
    // just conservatively return that event is not yet completed.
    std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
    auto HostVisibleEvent = Event->HostVisibleEvent;
    if (Event->Completed) {
      Result = PI_EVENT_COMPLETE;
    } else if (HostVisibleEvent) {
      ze_result_t ZeResult;
      ZeResult =
          ZE_CALL_NOCHECK(zeEventQueryStatus, (HostVisibleEvent->ZeEvent));
      if (ZeResult == ZE_RESULT_SUCCESS) {
        Result = PI_EVENT_COMPLETE;
      }
    }
    return ReturnValue(ur_cast<pi_int32>(Result));
  }
  case PI_EVENT_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Event->RefCount.load()});
  default:
    urPrint("Unsupported ParamName in piEventGetInfo: ParamName=%d(%x)\n",
            ParamName, ParamName);
    return PI_ERROR_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

pi_result piEventGetProfilingInfo(pi_event Event, pi_profiling_info ParamName,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {

  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);

  std::shared_lock<ur_shared_mutex> EventLock(Event->Mutex);
  if (Event->Queue &&
      (Event->Queue->Properties & PI_QUEUE_FLAG_PROFILING_ENABLE) == 0) {
    return PI_ERROR_PROFILING_INFO_NOT_AVAILABLE;
  }

  pi_device Device =
      Event->Queue ? Event->Queue->Device : Event->Context->Devices[0];

  uint64_t ZeTimerResolution = Device->ZeDeviceProperties->timerResolution;
  const uint64_t TimestampMaxValue =
      ((1ULL << Device->ZeDeviceProperties->kernelTimestampValidBits) - 1ULL);

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  ze_kernel_timestamp_result_t tsResult;

  switch (ParamName) {
  case PI_PROFILING_INFO_COMMAND_START: {
    ZE_CALL(zeEventQueryKernelTimestamp, (Event->ZeEvent, &tsResult));
    uint64_t ContextStartTime =
        (tsResult.global.kernelStart & TimestampMaxValue) * ZeTimerResolution;
    return ReturnValue(ContextStartTime);
  }
  case PI_PROFILING_INFO_COMMAND_END: {
    ZE_CALL(zeEventQueryKernelTimestamp, (Event->ZeEvent, &tsResult));

    uint64_t ContextStartTime =
        (tsResult.global.kernelStart & TimestampMaxValue);
    uint64_t ContextEndTime = (tsResult.global.kernelEnd & TimestampMaxValue);

    //
    // Handle a possible wrap-around (the underlying HW counter is < 64-bit).
    // Note, it will not report correct time if there were multiple wrap
    // arounds, and the longer term plan is to enlarge the capacity of the
    // HW timestamps.
    //
    if (ContextEndTime <= ContextStartTime) {
      ContextEndTime += TimestampMaxValue;
    }
    ContextEndTime *= ZeTimerResolution;
    return ReturnValue(ContextEndTime);
  }
  case PI_PROFILING_INFO_COMMAND_QUEUED:
  case PI_PROFILING_INFO_COMMAND_SUBMIT:
    // Note: No users for this case
    // The "command_submit" time is implemented by recording submission
    // timestamp with a call to piGetDeviceAndHostTimer before command enqueue.
    //
    return ReturnValue(uint64_t{0});
  default:
    urPrint("piEventGetProfilingInfo: not supported ParamName\n");
    return PI_ERROR_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

} // extern "C"

// Perform any necessary cleanup after an event has been signalled.
// This currently makes sure to release any kernel that may have been used by
// the event, updates the last command event in the queue and cleans up all dep
// events of the event.
// If the caller locks queue mutex then it must pass 'true' to QueueLocked.
static pi_result CleanupCompletedEvent(pi_event Event, bool QueueLocked) {
  pi_kernel AssociatedKernel = nullptr;
  // List of dependent events.
  std::list<pi_event> EventsToBeReleased;
  pi_queue AssociatedQueue = nullptr;
  {
    std::scoped_lock<ur_shared_mutex> EventLock(Event->Mutex);
    // Exit early of event was already cleanedup.
    if (Event->CleanedUp)
      return PI_SUCCESS;

    AssociatedQueue = Event->Queue;

    // Remember the kernel associated with this event if there is one. We are
    // going to release it later.
    if (Event->CommandType == PI_COMMAND_TYPE_NDRANGE_KERNEL &&
        Event->CommandData) {
      AssociatedKernel = ur_cast<pi_kernel>(Event->CommandData);
      Event->CommandData = nullptr;
    }

    // Make a list of all the dependent events that must have signalled
    // because this event was dependent on them.
    Event->WaitList.collectEventsForReleaseAndDestroyPiZeEventList(
        EventsToBeReleased);

    Event->CleanedUp = true;
  }

  auto ReleaseIndirectMem = [](pi_kernel Kernel) {
    if (IndirectAccessTrackingEnabled) {
      // piKernelRelease is called by CleanupCompletedEvent(Event) as soon as
      // kernel execution has finished. This is the place where we need to
      // release memory allocations. If kernel is not in use (not submitted by
      // some other thread) then release referenced memory allocations. As a
      // result, memory can be deallocated and context can be removed from
      // container in the platform. That's why we need to lock a mutex here.
      pi_platform Plt = Kernel->Program->Context->getPlatform();
      std::scoped_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex);

      if (--Kernel->SubmissionsCount == 0) {
        // Kernel is not submitted for execution, release referenced memory
        // allocations.
        for (auto &MemAlloc : Kernel->MemAllocs) {
          // std::pair<void *const, MemAllocRecord> *, Hash
          USMFreeHelper(MemAlloc->second.Context, MemAlloc->first,
                        MemAlloc->second.OwnZeMemHandle);
        }
        Kernel->MemAllocs.clear();
      }
    }
  };

  // We've reset event data members above, now cleanup resources.
  if (AssociatedKernel) {
    ReleaseIndirectMem(AssociatedKernel);
    PI_CALL(piKernelRelease(AssociatedKernel));
  }

  if (AssociatedQueue) {
    {
      // Lock automatically releases when this goes out of scope.
      std::unique_lock<ur_shared_mutex> QueueLock(AssociatedQueue->Mutex,
                                                  std::defer_lock);
      if (!QueueLocked)
        QueueLock.lock();

      // If this event was the LastCommandEvent in the queue, being used
      // to make sure that commands were executed in-order, remove this.
      // If we don't do this, the event can get released and freed leaving
      // a dangling pointer to this event.  It could also cause unneeded
      // already finished events to show up in the wait list.
      if (AssociatedQueue->LastCommandEvent == Event) {
        AssociatedQueue->LastCommandEvent = nullptr;
      }
    }

    // Release this event since we explicitly retained it on creation and
    // association with queue. Events which don't have associated queue doesn't
    // require this release because it means that they are not created using
    // createEventAndAssociateQueue, i.e. additional retain was not made.
    PI_CALL(piEventReleaseInternal(Event));
  }

  // The list of dependent events will be appended to as we walk it so that this
  // algorithm doesn't go recursive due to dependent events themselves being
  // dependent on other events forming a potentially very deep tree, and deep
  // recursion.  That turned out to be a significant problem with the recursive
  // code that preceded this implementation.
  while (!EventsToBeReleased.empty()) {
    pi_event DepEvent = EventsToBeReleased.front();
    DepEvent->Completed = true;
    EventsToBeReleased.pop_front();

    pi_kernel DepEventKernel = nullptr;
    {
      std::scoped_lock<ur_shared_mutex> DepEventLock(DepEvent->Mutex);
      DepEvent->WaitList.collectEventsForReleaseAndDestroyPiZeEventList(
          EventsToBeReleased);
      if (IndirectAccessTrackingEnabled) {
        // DepEvent has finished, we can release the associated kernel if there
        // is one. This is the earliest place we can do this and it can't be
        // done twice, so it is safe. Lock automatically releases when this goes
        // out of scope.
        // TODO: this code needs to be moved out of the guard.
        if (DepEvent->CommandType == PI_COMMAND_TYPE_NDRANGE_KERNEL &&
            DepEvent->CommandData) {
          DepEventKernel = ur_cast<pi_kernel>(DepEvent->CommandData);
          DepEvent->CommandData = nullptr;
        }
      }
    }
    if (DepEventKernel) {
      ReleaseIndirectMem(DepEventKernel);
      PI_CALL(piKernelRelease(DepEventKernel));
    }
    PI_CALL(piEventReleaseInternal(DepEvent));
  }

  return PI_SUCCESS;
}

extern "C" {

pi_result piEventsWait(pi_uint32 NumEvents, const pi_event *EventList) {

  if (NumEvents && !EventList) {
    return PI_ERROR_INVALID_EVENT;
  }
  for (uint32_t I = 0; I < NumEvents; I++) {
    if (EventList[I]->Queue->Device->ZeEventsScope ==
        OnDemandHostVisibleProxy) {
      // Make sure to add all host-visible "proxy" event signals if needed.
      // This ensures that all signalling commands are submitted below and
      // thus proxy events can be waited without a deadlock.
      //
      if (!EventList[I]->hasExternalRefs())
        die("piEventsWait must not be called for an internal event");

      ze_event_handle_t ZeHostVisibleEvent;
      if (auto Res =
              EventList[I]->getOrCreateHostVisibleEvent(ZeHostVisibleEvent))
        return Res;
    }
  }
  // Submit dependent open command lists for execution, if any
  for (uint32_t I = 0; I < NumEvents; I++) {
    auto Queue = EventList[I]->Queue;
    if (Queue) {
      // Lock automatically releases when this goes out of scope.
      std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

      if (auto Res = Queue->executeAllOpenCommandLists())
        return Res;
    }
  }
  std::unordered_set<pi_queue> Queues;
  for (uint32_t I = 0; I < NumEvents; I++) {
    {
      {
        std::shared_lock<ur_shared_mutex> EventLock(EventList[I]->Mutex);
        if (!EventList[I]->hasExternalRefs())
          die("piEventsWait must not be called for an internal event");

        if (!EventList[I]->Completed) {
          auto HostVisibleEvent = EventList[I]->HostVisibleEvent;
          if (!HostVisibleEvent)
            die("The host-visible proxy event missing");

          ze_event_handle_t ZeEvent = HostVisibleEvent->ZeEvent;
          urPrint("ZeEvent = %#llx\n", ur_cast<std::uintptr_t>(ZeEvent));
          ZE_CALL(zeHostSynchronize, (ZeEvent));
          EventList[I]->Completed = true;
        }
      }
      if (auto Q = EventList[I]->Queue) {
        if (Q->UsingImmCmdLists && Q->isInOrderQueue())
          // Use information about waited event to cleanup completed events in
          // the in-order queue.
          CleanupEventsInImmCmdLists(EventList[I]->Queue,
                                     /* QueueLocked */ false,
                                     /* QueueSynced */ false, EventList[I]);
        else {
          // NOTE: we are cleaning up after the event here to free resources
          // sooner in case run-time is not calling piEventRelease soon enough.
          CleanupCompletedEvent(EventList[I]);
          // For the case when we have out-of-order queue or regular command
          // lists its more efficient to check fences so put the queue in the
          // set to cleanup later.
          Queues.insert(Q);
        }
      }
    }
  }

  // We waited some events above, check queue for signaled command lists and
  // reset them.
  for (auto &Q : Queues) {
    std::unique_lock<ur_shared_mutex> Lock(Q->Mutex);
    resetCommandLists(Q);
  }
  return PI_SUCCESS;
}

pi_result piEventSetCallback(pi_event Event, pi_int32 CommandExecCallbackType,
                             void (*PFnNotify)(pi_event Event,
                                               pi_int32 EventCommandStatus,
                                               void *UserData),
                             void *UserData) {
  (void)Event;
  (void)CommandExecCallbackType;
  (void)PFnNotify;
  (void)UserData;
  die("piEventSetCallback: deprecated, to be removed");
  return PI_SUCCESS;
}

pi_result piEventSetStatus(pi_event Event, pi_int32 ExecutionStatus) {
  (void)Event;
  (void)ExecutionStatus;
  die("piEventSetStatus: deprecated, to be removed");
  return PI_SUCCESS;
}

pi_result piEventRetain(pi_event Event) {
  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);
  Event->RefCountExternal++;
  Event->RefCount.increment();
  return PI_SUCCESS;
}

pi_result piEventRelease(pi_event Event) {
  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);
  Event->RefCountExternal--;
  PI_CALL(piEventReleaseInternal(Event));
  return PI_SUCCESS;
}

void _pi_queue::active_barriers::add(pi_event &Event) {
  Event->RefCount.increment();
  Events.push_back(Event);
}

pi_result _pi_queue::active_barriers::clear() {
  for (const auto &Event : Events)
    PI_CALL(piEventReleaseInternal(Event));
  Events.clear();
  return PI_SUCCESS;
}

static pi_result piEventReleaseInternal(pi_event Event) {
  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);

  if (!Event->RefCount.decrementAndTest())
    return PI_SUCCESS;

  if (Event->CommandType == PI_COMMAND_TYPE_MEM_BUFFER_UNMAP &&
      Event->CommandData) {
    // Free the memory allocated in the piEnqueueMemBufferMap.
    if (auto Res = ZeMemFreeHelper(Event->Context, Event->CommandData))
      return Res;
    Event->CommandData = nullptr;
  }
  if (Event->OwnZeEvent) {
    if (DisableEventsCaching) {
      auto ZeResult = ZE_CALL_NOCHECK(zeEventDestroy, (Event->ZeEvent));
      // Gracefully handle the case that L0 was already unloaded.
      if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
        return mapError(ZeResult);

      auto Context = Event->Context;
      if (auto Res = Context->decrementUnreleasedEventsInPool(Event))
        return Res;
    }
  }
  // It is possible that host-visible event was never created.
  // In case it was check if that's different from this same event
  // and release a reference to it.
  if (Event->HostVisibleEvent && Event->HostVisibleEvent != Event) {
    // Decrement ref-count of the host-visible proxy event.
    PI_CALL(piEventReleaseInternal(Event->HostVisibleEvent));
  }

  // Save pointer to the queue before deleting/resetting event.
  // When we add an event to the cache we need to check whether profiling is
  // enabled or not, so we access properties of the queue and that's why queue
  // must released later.
  auto Queue = Event->Queue;
  if (DisableEventsCaching || !Event->OwnZeEvent) {
    delete Event;
  } else {
    Event->Context->addEventToContextCache(Event);
  }

  // We intentionally incremented the reference counter when an event is
  // created so that we can avoid pi_queue is released before the associated
  // pi_event is released. Here we have to decrement it so pi_queue
  // can be released successfully.
  if (Queue) {
    PI_CALL(piQueueReleaseInternal(Queue));
  }

  return PI_SUCCESS;
}

pi_result piextEventGetNativeHandle(pi_event Event,
                                    pi_native_handle *NativeHandle) {
  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  {
    std::shared_lock<ur_shared_mutex> Lock(Event->Mutex);
    auto *ZeEvent = ur_cast<ze_event_handle_t *>(NativeHandle);
    *ZeEvent = Event->ZeEvent;
  }
  // Event can potentially be in an open command-list, make sure that
  // it is submitted for execution to avoid potential deadlock if
  // interop app is going to wait for it.
  auto Queue = Event->Queue;
  if (Queue) {
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);
    const auto &OpenCommandList = Queue->eventOpenCommandList(Event);
    if (OpenCommandList != Queue->CommandListMap.end()) {
      if (auto Res = Queue->executeOpenCommandList(
              OpenCommandList->second.isCopy(Queue)))
        return Res;
    }
  }
  return PI_SUCCESS;
}

pi_result piextEventCreateWithNativeHandle(pi_native_handle NativeHandle,
                                           pi_context Context,
                                           bool OwnNativeHandle,
                                           pi_event *Event) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Event, PI_ERROR_INVALID_EVENT);
  PI_ASSERT(NativeHandle, PI_ERROR_INVALID_VALUE);

  auto ZeEvent = ur_cast<ze_event_handle_t>(NativeHandle);
  *Event = new _pi_event(ZeEvent, nullptr /* ZeEventPool */, Context,
                         PI_COMMAND_TYPE_USER, OwnNativeHandle);

  // Assume native event is host-visible, or otherwise we'd
  // need to create a host-visible proxy for it.
  (*Event)->HostVisibleEvent = *Event;

  // Unlike regular events managed by SYCL RT we don't have to wait for interop
  // events completion, and not need to do the their `cleanup()`. This in
  // particular guarantees that the extra `piEventRelease` is not called on
  // them. That release is needed to match the `piEventRetain` of regular events
  // made for waiting for event completion, but not this interop event.
  (*Event)->CleanedUp = true;

  return PI_SUCCESS;
}

//
// Sampler
//
pi_result piSamplerCreate(pi_context Context,
                          const pi_sampler_properties *SamplerProperties,
                          pi_sampler *RetSampler) {

  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(RetSampler, PI_ERROR_INVALID_VALUE);

  std::shared_lock<ur_shared_mutex> Lock(Context->Mutex);

  // Have the "0" device in context to own the sampler. Rely on Level-Zero
  // drivers to perform migration as necessary for sharing it across multiple
  // devices in the context.
  //
  // TODO: figure out if we instead need explicit copying for acessing
  // the sampler from other devices in the context.
  //
  pi_device Device = Context->Devices[0];

  ze_sampler_handle_t ZeSampler;
  ZeStruct<ze_sampler_desc_t> ZeSamplerDesc;

  // Set the default values for the ZeSamplerDesc.
  ZeSamplerDesc.isNormalized = PI_TRUE;
  ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP;
  ZeSamplerDesc.filterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;

  // Update the values of the ZeSamplerDesc from the pi_sampler_properties list.
  // Default values will be used if any of the following is true:
  //   a) SamplerProperties list is NULL
  //   b) SamplerProperties list is missing any properties

  if (SamplerProperties) {
    const pi_sampler_properties *CurProperty = SamplerProperties;

    while (*CurProperty != 0) {
      switch (*CurProperty) {
      case PI_SAMPLER_PROPERTIES_NORMALIZED_COORDS: {
        pi_bool CurValueBool = ur_cast<pi_bool>(*(++CurProperty));

        if (CurValueBool == PI_TRUE)
          ZeSamplerDesc.isNormalized = PI_TRUE;
        else if (CurValueBool == PI_FALSE)
          ZeSamplerDesc.isNormalized = PI_FALSE;
        else {
          urPrint("piSamplerCreate: unsupported "
                  "PI_SAMPLER_NORMALIZED_COORDS value\n");
          return PI_ERROR_INVALID_VALUE;
        }
      } break;

      case PI_SAMPLER_PROPERTIES_ADDRESSING_MODE: {
        pi_sampler_addressing_mode CurValueAddressingMode =
            ur_cast<pi_sampler_addressing_mode>(
                ur_cast<pi_uint32>(*(++CurProperty)));

        // Level Zero runtime with API version 1.2 and lower has a bug:
        // ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER is implemented as "clamp to
        // edge" and ZE_SAMPLER_ADDRESS_MODE_CLAMP is implemented as "clamp to
        // border", i.e. logic is flipped. Starting from API version 1.3 this
        // problem is going to be fixed. That's why check for API version to set
        // an address mode.
        ze_api_version_t ZeApiVersion = Context->getPlatform()->ZeApiVersion;
        // TODO: add support for PI_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE
        switch (CurValueAddressingMode) {
        case PI_SAMPLER_ADDRESSING_MODE_NONE:
          ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
          break;
        case PI_SAMPLER_ADDRESSING_MODE_REPEAT:
          ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_REPEAT;
          break;
        case PI_SAMPLER_ADDRESSING_MODE_CLAMP:
          ZeSamplerDesc.addressMode =
              ZeApiVersion < ZE_MAKE_VERSION(1, 3)
                  ? ZE_SAMPLER_ADDRESS_MODE_CLAMP
                  : ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
          break;
        case PI_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE:
          ZeSamplerDesc.addressMode =
              ZeApiVersion < ZE_MAKE_VERSION(1, 3)
                  ? ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
                  : ZE_SAMPLER_ADDRESS_MODE_CLAMP;
          break;
        case PI_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT:
          ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_MIRROR;
          break;
        default:
          urPrint("piSamplerCreate: unsupported PI_SAMPLER_ADDRESSING_MODE "
                  "value\n");
          urPrint("PI_SAMPLER_ADDRESSING_MODE=%d\n", CurValueAddressingMode);
          return PI_ERROR_INVALID_VALUE;
        }
      } break;

      case PI_SAMPLER_PROPERTIES_FILTER_MODE: {
        pi_sampler_filter_mode CurValueFilterMode =
            ur_cast<pi_sampler_filter_mode>(
                ur_cast<pi_uint32>(*(++CurProperty)));

        if (CurValueFilterMode == PI_SAMPLER_FILTER_MODE_NEAREST)
          ZeSamplerDesc.filterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
        else if (CurValueFilterMode == PI_SAMPLER_FILTER_MODE_LINEAR)
          ZeSamplerDesc.filterMode = ZE_SAMPLER_FILTER_MODE_LINEAR;
        else {
          urPrint("PI_SAMPLER_FILTER_MODE=%d\n", CurValueFilterMode);
          urPrint(
              "piSamplerCreate: unsupported PI_SAMPLER_FILTER_MODE value\n");
          return PI_ERROR_INVALID_VALUE;
        }
      } break;

      default:
        break;
      }
      CurProperty++;
    }
  }

  ZE_CALL(zeSamplerCreate, (Context->ZeContext, Device->ZeDevice,
                            &ZeSamplerDesc, // TODO: translate properties
                            &ZeSampler));

  try {
    *RetSampler = new _pi_sampler(ZeSampler);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

pi_result piSamplerGetInfo(pi_sampler Sampler, pi_sampler_info ParamName,
                           size_t ParamValueSize, void *ParamValue,
                           size_t *ParamValueSizeRet) {
  (void)Sampler;
  (void)ParamName;
  (void)ParamValueSize;
  (void)ParamValue;
  (void)ParamValueSizeRet;

  die("piSamplerGetInfo: not implemented");
  return {};
}

pi_result piSamplerRetain(pi_sampler Sampler) {
  PI_ASSERT(Sampler, PI_ERROR_INVALID_SAMPLER);

  Sampler->RefCount.increment();
  return PI_SUCCESS;
}

pi_result piSamplerRelease(pi_sampler Sampler) {
  PI_ASSERT(Sampler, PI_ERROR_INVALID_SAMPLER);

  if (!Sampler->RefCount.decrementAndTest())
    return PI_SUCCESS;

  auto ZeResult = ZE_CALL_NOCHECK(zeSamplerDestroy, (Sampler->ZeSampler));
  // Gracefully handle the case that L0 was already unloaded.
  if (ZeResult && ZeResult != ZE_RESULT_ERROR_UNINITIALIZED)
    return mapError(ZeResult);

  delete Sampler;
  return PI_SUCCESS;
}

//
// Queue Commands
//
pi_result piEnqueueEventsWait(pi_queue Queue, pi_uint32 NumEventsInWaitList,
                              const pi_event *EventWaitList,
                              pi_event *OutEvent) {

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  if (EventWaitList) {
    PI_ASSERT(NumEventsInWaitList > 0, PI_ERROR_INVALID_VALUE);

    bool UseCopyEngine = false;

    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

    _pi_ze_event_list_t TmpWaitList = {};
    if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
            NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine))
      return Res;

    // Get a new command list to be used on this call
    pi_command_list_ptr_t CommandList{};
    if (auto Res = Queue->Context->getAvailableCommandList(Queue, CommandList,
                                                           UseCopyEngine))
      return Res;

    ze_event_handle_t ZeEvent = nullptr;
    pi_event InternalEvent;
    bool IsInternal = OutEvent == nullptr;
    pi_event *Event = OutEvent ? OutEvent : &InternalEvent;
    auto Res = createEventAndAssociateQueue(Queue, Event, PI_COMMAND_TYPE_USER,
                                            CommandList, IsInternal);
    if (Res != PI_SUCCESS)
      return Res;

    ZeEvent = (*Event)->ZeEvent;
    (*Event)->WaitList = TmpWaitList;

    const auto &WaitList = (*Event)->WaitList;
    auto ZeCommandList = CommandList->first;
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));

    ZE_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

    // Execute command list asynchronously as the event will be used
    // to track down its completion.
    return Queue->executeCommandList(CommandList);
  }

  {
    // If wait-list is empty, then this particular command should wait until
    // all previous enqueued commands to the command-queue have completed.
    //
    // TODO: find a way to do that without blocking the host.

    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

    if (OutEvent) {
      auto Res = createEventAndAssociateQueue(
          Queue, OutEvent, PI_COMMAND_TYPE_USER, Queue->CommandListMap.end(),
          /* IsInternal */ false);
      if (Res != PI_SUCCESS)
        return Res;
    }

    Queue->synchronize();

    if (OutEvent) {
      Queue->LastCommandEvent = *OutEvent;

      ZE_CALL(zeEventHostSignal, ((*OutEvent)->ZeEvent));
      (*OutEvent)->Completed = true;
    }
  }

  if (!Queue->UsingImmCmdLists) {
    std::unique_lock<ur_shared_mutex> Lock(Queue->Mutex);
    resetCommandLists(Queue);
  }

  return PI_SUCCESS;
}

pi_result piEnqueueEventsWaitWithBarrier(pi_queue Queue,
                                         pi_uint32 NumEventsInWaitList,
                                         const pi_event *EventWaitList,
                                         pi_event *OutEvent) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  // Helper function for appending a barrier to a command list.
  auto insertBarrierIntoCmdList =
      [&Queue](pi_command_list_ptr_t CmdList,
               const _pi_ze_event_list_t &EventWaitList, pi_event &Event,
               bool IsInternal) {
        if (auto Res = createEventAndAssociateQueue(
                Queue, &Event, PI_COMMAND_TYPE_USER, CmdList, IsInternal))
          return Res;

        Event->WaitList = EventWaitList;
        ZE_CALL(zeCommandListAppendBarrier,
                (CmdList->first, Event->ZeEvent, EventWaitList.Length,
                 EventWaitList.ZeEventList));
        return PI_SUCCESS;
      };

  pi_event InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  pi_event *Event = OutEvent ? OutEvent : &InternalEvent;

  // Indicator for whether batching is allowed. This may be changed later in
  // this function, but allow it by default.
  bool OkToBatch = true;

  // If we have a list of events to make the barrier from, then we can create a
  // barrier on these and use the resulting event as our future barrier.
  // We use the same approach if
  // UR_L0_USE_MULTIPLE_COMMANDLIST_BARRIERS is not set to a
  // positive value.
  // We use the same approach if we have in-order queue because every command
  // depends on previous one, so we don't need to insert barrier to multiple
  // command lists.
  if (NumEventsInWaitList || !UseMultipleCmdlistBarriers ||
      Queue->isInOrderQueue()) {
    // Retain the events as they will be owned by the result event.
    _pi_ze_event_list_t TmpWaitList;
    if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
            NumEventsInWaitList, EventWaitList, Queue,
            /*UseCopyEngine=*/false))
      return Res;

    // Get an arbitrary command-list in the queue.
    pi_command_list_ptr_t CmdList;
    if (auto Res = Queue->Context->getAvailableCommandList(
            Queue, CmdList,
            /*UseCopyEngine=*/false, OkToBatch))
      return Res;

    // Insert the barrier into the command-list and execute.
    if (auto Res =
            insertBarrierIntoCmdList(CmdList, TmpWaitList, *Event, IsInternal))
      return Res;

    if (auto Res = Queue->executeCommandList(CmdList, false, OkToBatch))
      return Res;

    // Because of the dependency between commands in the in-order queue we don't
    // need to keep track of any active barriers if we have in-order queue.
    if (UseMultipleCmdlistBarriers && !Queue->isInOrderQueue()) {
      Queue->ActiveBarriers.add(*Event);
    }
    return PI_SUCCESS;
  }

  // Since there are no events to explicitly create a barrier for, we are
  // inserting a queue-wide barrier.

  // Command list(s) for putting barriers.
  std::vector<pi_command_list_ptr_t> CmdLists;

  // There must be at least one L0 queue.
  auto &ComputeGroup = Queue->ComputeQueueGroupsByTID.get();
  auto &CopyGroup = Queue->CopyQueueGroupsByTID.get();
  PI_ASSERT(!ComputeGroup.ZeQueues.empty() || !CopyGroup.ZeQueues.empty(),
            PI_ERROR_INVALID_QUEUE);

  size_t NumQueues = 0;
  for (auto &QueueMap :
       {Queue->ComputeQueueGroupsByTID, Queue->CopyQueueGroupsByTID})
    for (auto &QueueGroup : QueueMap)
      NumQueues += QueueGroup.second.ZeQueues.size();

  OkToBatch = true;
  // Get an available command list tied to each command queue. We need
  // these so a queue-wide barrier can be inserted into each command
  // queue.
  CmdLists.reserve(NumQueues);
  for (auto &QueueMap :
       {Queue->ComputeQueueGroupsByTID, Queue->CopyQueueGroupsByTID})
    for (auto &QueueGroup : QueueMap) {
      bool UseCopyEngine =
          QueueGroup.second.Type != _pi_queue::queue_type::Compute;
      if (Queue->UsingImmCmdLists) {
        // If immediate command lists are being used, each will act as their own
        // queue, so we must insert a barrier into each.
        for (auto &ImmCmdList : QueueGroup.second.ImmCmdLists)
          if (ImmCmdList != Queue->CommandListMap.end())
            CmdLists.push_back(ImmCmdList);
      } else {
        for (auto ZeQueue : QueueGroup.second.ZeQueues) {
          if (ZeQueue) {
            pi_command_list_ptr_t CmdList;
            if (auto Res = Queue->Context->getAvailableCommandList(
                    Queue, CmdList, UseCopyEngine, OkToBatch, &ZeQueue))
              return Res;
            CmdLists.push_back(CmdList);
          }
        }
      }
    }

  // If no activity has occurred on the queue then there will be no cmdlists.
  // We need one for generating an Event, so create one.
  if (CmdLists.size() == 0) {
    // Get any available command list.
    pi_command_list_ptr_t CmdList;
    if (auto Res = Queue->Context->getAvailableCommandList(
            Queue, CmdList,
            /*UseCopyEngine=*/false, OkToBatch))
      return Res;
    CmdLists.push_back(CmdList);
  }

  if (CmdLists.size() > 1) {
    // Insert a barrier into each unique command queue using the available
    // command-lists.
    std::vector<pi_event> EventWaitVector(CmdLists.size());
    for (size_t I = 0; I < CmdLists.size(); ++I) {
      if (auto Res =
              insertBarrierIntoCmdList(CmdLists[I], _pi_ze_event_list_t{},
                                       EventWaitVector[I], /*IsInternal*/ true))
        return Res;
    }
    // If there were multiple queues we need to create a "convergence" event to
    // be our active barrier. This convergence event is signalled by a barrier
    // on all the events from the barriers we have inserted into each queue.
    // Use the first command list as our convergence command list.
    pi_command_list_ptr_t &ConvergenceCmdList = CmdLists[0];

    // Create an event list. It will take ownership over all relevant events so
    // we relinquish ownership and let it keep all events it needs.
    _pi_ze_event_list_t BaseWaitList;
    if (auto Res = BaseWaitList.createAndRetainPiZeEventList(
            EventWaitVector.size(), EventWaitVector.data(), Queue,
            ConvergenceCmdList->second.isCopy(Queue)))
      return Res;

    // Insert a barrier with the events from each command-queue into the
    // convergence command list. The resulting event signals the convergence of
    // all barriers.
    if (auto Res = insertBarrierIntoCmdList(ConvergenceCmdList, BaseWaitList,
                                            *Event, IsInternal))
      return Res;
  } else {
    // If there is only a single queue then insert a barrier and the single
    // result event can be used as our active barrier and used as the return
    // event. Take into account whether output event is discarded or not.
    if (auto Res = insertBarrierIntoCmdList(CmdLists[0], _pi_ze_event_list_t{},
                                            *Event, IsInternal))
      return Res;
  }

  // Execute each command list so the barriers can be encountered.
  for (pi_command_list_ptr_t &CmdList : CmdLists)
    if (auto Res = Queue->executeCommandList(CmdList, false, OkToBatch))
      return Res;

  if (auto Res = Queue->ActiveBarriers.clear())
    return Res;
  Queue->ActiveBarriers.add(*Event);
  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferRead(pi_queue Queue, pi_mem Src,
                                 pi_bool BlockingRead, size_t Offset,
                                 size_t Size, void *Dst,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {
  PI_ASSERT(Src, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  std::shared_lock<ur_shared_mutex> SrcLock(Src->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex> LockAll(
      SrcLock, Queue->Mutex);

  char *ZeHandleSrc;
  PI_CALL(Src->getZeHandle(ZeHandleSrc, _pi_mem::read_only, Queue->Device));
  return enqueueMemCopyHelper(PI_COMMAND_TYPE_MEM_BUFFER_READ, Queue, Dst,
                              BlockingRead, Size, ZeHandleSrc + Offset,
                              NumEventsInWaitList, EventWaitList, Event,
                              /* PreferCopyEngine */ true);
}

pi_result piEnqueueMemBufferReadRect(
    pi_queue Queue, pi_mem Buffer, pi_bool BlockingRead,
    pi_buff_rect_offset BufferOffset, pi_buff_rect_offset HostOffset,
    pi_buff_rect_region Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, void *Ptr,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *Event) {

  PI_ASSERT(Buffer, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  std::shared_lock<ur_shared_mutex> SrcLock(Buffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex> LockAll(
      SrcLock, Queue->Mutex);

  char *ZeHandleSrc;
  PI_CALL(Buffer->getZeHandle(ZeHandleSrc, _pi_mem::read_only, Queue->Device));
  return enqueueMemCopyRectHelper(
      PI_COMMAND_TYPE_MEM_BUFFER_READ_RECT, Queue, ZeHandleSrc,
      static_cast<char *>(Ptr), BufferOffset, HostOffset, Region,
      BufferRowPitch, HostRowPitch, BufferSlicePitch, HostSlicePitch,
      BlockingRead, NumEventsInWaitList, EventWaitList, Event);
}

} // extern "C"

bool _pi_queue::useCopyEngine(bool PreferCopyEngine) const {
  auto InitialCopyGroup = CopyQueueGroupsByTID.begin()->second;
  return PreferCopyEngine && InitialCopyGroup.ZeQueues.size() > 0 &&
         (!isInOrderQueue() || UseCopyEngineForInOrderQueue);
}

// Wait on all operations in flight on this Queue.
// The caller is expected to hold a lock on the Queue.
// For standard commandlists sync the L0 queues directly.
// For immediate commandlists add barriers to all commandlists associated
// with the Queue. An alternative approach would be to wait on all Events
// associated with the in-flight operations.
// TODO: Event release in immediate commandlist mode is driven by the SYCL
// runtime. Need to investigate whether relase can be done earlier, at sync
// points such as this, to reduce total number of active Events.
pi_result _pi_queue::synchronize() {
  if (!Healthy)
    return PI_SUCCESS;

  auto syncImmCmdList = [](_pi_queue *Queue, pi_command_list_ptr_t ImmCmdList) {
    if (ImmCmdList == Queue->CommandListMap.end())
      return PI_SUCCESS;

    pi_event Event;
    pi_result Res =
        createEventAndAssociateQueue(Queue, &Event, PI_COMMAND_TYPE_USER,
                                     ImmCmdList, /* IsInternal */ false);
    if (Res != PI_SUCCESS)
      return Res;
    auto zeEvent = Event->ZeEvent;
    ZE_CALL(zeCommandListAppendBarrier,
            (ImmCmdList->first, zeEvent, 0, nullptr));
    ZE_CALL(zeHostSynchronize, (zeEvent));
    Event->Completed = true;
    PI_CALL(piEventRelease(Event));

    // Cleanup all events from the synced command list.
    auto EventListToCleanup = std::move(ImmCmdList->second.EventList);
    ImmCmdList->second.EventList.clear();
    CleanupEventListFromResetCmdList(EventListToCleanup, true);
    return PI_SUCCESS;
  };

  if (LastCommandEvent) {
    // For in-order queue just wait for the last command.
    // If event is discarded then it can be in reset state or underlying level
    // zero handle can have device scope, so we can't synchronize the last
    // event.
    if (isInOrderQueue() && !LastCommandEvent->IsDiscarded) {
      ZE_CALL(zeHostSynchronize, (LastCommandEvent->ZeEvent));
    } else {
      // Otherwise sync all L0 queues/immediate command-lists.
      for (auto &QueueMap : {ComputeQueueGroupsByTID, CopyQueueGroupsByTID}) {
        for (auto &QueueGroup : QueueMap) {
          if (UsingImmCmdLists) {
            for (auto ImmCmdList : QueueGroup.second.ImmCmdLists)
              syncImmCmdList(this, ImmCmdList);
          } else {
            for (auto &ZeQueue : QueueGroup.second.ZeQueues)
              if (ZeQueue)
                ZE_CALL(zeHostSynchronize, (ZeQueue));
          }
        }
      }
    }
    LastCommandEvent = nullptr;
  }
  // With the entire queue synchronized, the active barriers must be done so we
  // can remove them.
  if (auto Res = ActiveBarriers.clear())
    return Res;

  return PI_SUCCESS;
}

// Shared by all memory read/write/copy PI interfaces.
// PI interfaces must have queue's and destination buffer's mutexes locked for
// exclusive use and source buffer's mutex locked for shared use on entry.
static pi_result
enqueueMemCopyHelper(pi_command_type CommandType, pi_queue Queue, void *Dst,
                     pi_bool BlockingWrite, size_t Size, const void *Src,
                     pi_uint32 NumEventsInWaitList,
                     const pi_event *EventWaitList, pi_event *OutEvent,
                     bool PreferCopyEngine) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);

  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
          NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine))
    return Res;

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  // Get a new command list to be used on this call
  pi_command_list_ptr_t CommandList{};
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, CommandList, UseCopyEngine, OkToBatch))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  pi_event InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  pi_event *Event = OutEvent ? OutEvent : &InternalEvent;
  auto Res = createEventAndAssociateQueue(Queue, Event, CommandType,
                                          CommandList, IsInternal);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  urPrint("calling zeCommandListAppendMemoryCopy() with\n"
          "  ZeEvent %#llx\n",
          ur_cast<std::uintptr_t>(ZeEvent));
  printZeEventList(WaitList);

  ZE_CALL(zeCommandListAppendMemoryCopy,
          (ZeCommandList, Dst, Src, Size, ZeEvent, WaitList.Length,
           WaitList.ZeEventList));

  if (auto Res =
          Queue->executeCommandList(CommandList, BlockingWrite, OkToBatch))
    return Res;

  return PI_SUCCESS;
}

// Shared by all memory read/write/copy rect PI interfaces.
// PI interfaces must have queue's and destination buffer's mutexes locked for
// exclusive use and source buffer's mutex locked for shared use on entry.
static pi_result enqueueMemCopyRectHelper(
    pi_command_type CommandType, pi_queue Queue, const void *SrcBuffer,
    void *DstBuffer, pi_buff_rect_offset SrcOrigin,
    pi_buff_rect_offset DstOrigin, pi_buff_rect_region Region,
    size_t SrcRowPitch, size_t DstRowPitch, size_t SrcSlicePitch,
    size_t DstSlicePitch, pi_bool Blocking, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *OutEvent, bool PreferCopyEngine) {

  PI_ASSERT(Region && SrcOrigin && DstOrigin && Queue, PI_ERROR_INVALID_VALUE);

  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);

  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
          NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine))
    return Res;

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  // Get a new command list to be used on this call
  pi_command_list_ptr_t CommandList{};
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, CommandList, UseCopyEngine, OkToBatch))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  pi_event InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  pi_event *Event = OutEvent ? OutEvent : &InternalEvent;
  auto Res = createEventAndAssociateQueue(Queue, Event, CommandType,
                                          CommandList, IsInternal);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  urPrint("calling zeCommandListAppendMemoryCopy() with\n"
          "  ZeEvent %#llx\n",
          ur_cast<std::uintptr_t>(ZeEvent));
  printZeEventList(WaitList);

  uint32_t SrcOriginX = ur_cast<uint32_t>(SrcOrigin->x_bytes);
  uint32_t SrcOriginY = ur_cast<uint32_t>(SrcOrigin->y_scalar);
  uint32_t SrcOriginZ = ur_cast<uint32_t>(SrcOrigin->z_scalar);

  uint32_t SrcPitch = SrcRowPitch;
  if (SrcPitch == 0)
    SrcPitch = ur_cast<uint32_t>(Region->width_bytes);

  if (SrcSlicePitch == 0)
    SrcSlicePitch = ur_cast<uint32_t>(Region->height_scalar) * SrcPitch;

  uint32_t DstOriginX = ur_cast<uint32_t>(DstOrigin->x_bytes);
  uint32_t DstOriginY = ur_cast<uint32_t>(DstOrigin->y_scalar);
  uint32_t DstOriginZ = ur_cast<uint32_t>(DstOrigin->z_scalar);

  uint32_t DstPitch = DstRowPitch;
  if (DstPitch == 0)
    DstPitch = ur_cast<uint32_t>(Region->width_bytes);

  if (DstSlicePitch == 0)
    DstSlicePitch = ur_cast<uint32_t>(Region->height_scalar) * DstPitch;

  uint32_t Width = ur_cast<uint32_t>(Region->width_bytes);
  uint32_t Height = ur_cast<uint32_t>(Region->height_scalar);
  uint32_t Depth = ur_cast<uint32_t>(Region->depth_scalar);

  const ze_copy_region_t ZeSrcRegion = {SrcOriginX, SrcOriginY, SrcOriginZ,
                                        Width,      Height,     Depth};
  const ze_copy_region_t ZeDstRegion = {DstOriginX, DstOriginY, DstOriginZ,
                                        Width,      Height,     Depth};

  ZE_CALL(zeCommandListAppendMemoryCopyRegion,
          (ZeCommandList, DstBuffer, &ZeDstRegion, DstPitch, DstSlicePitch,
           SrcBuffer, &ZeSrcRegion, SrcPitch, SrcSlicePitch, nullptr,
           WaitList.Length, WaitList.ZeEventList));

  urPrint("calling zeCommandListAppendMemoryCopyRegion()\n");

  ZE_CALL(zeCommandListAppendBarrier, (ZeCommandList, ZeEvent, 0, nullptr));

  urPrint("calling zeCommandListAppendBarrier() with Event %#llx\n",
          ur_cast<std::uintptr_t>(ZeEvent));

  if (auto Res = Queue->executeCommandList(CommandList, Blocking, OkToBatch))
    return Res;

  return PI_SUCCESS;
}

extern "C" {

pi_result piEnqueueMemBufferWrite(pi_queue Queue, pi_mem Buffer,
                                  pi_bool BlockingWrite, size_t Offset,
                                  size_t Size, const void *Ptr,
                                  pi_uint32 NumEventsInWaitList,
                                  const pi_event *EventWaitList,
                                  pi_event *Event) {

  PI_ASSERT(Buffer, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  char *ZeHandleDst;
  PI_CALL(Buffer->getZeHandle(ZeHandleDst, _pi_mem::write_only, Queue->Device));
  return enqueueMemCopyHelper(PI_COMMAND_TYPE_MEM_BUFFER_WRITE, Queue,
                              ZeHandleDst + Offset, // dst
                              BlockingWrite, Size,
                              Ptr, // src
                              NumEventsInWaitList, EventWaitList, Event,
                              /* PreferCopyEngine */ true);
}

pi_result piEnqueueMemBufferWriteRect(
    pi_queue Queue, pi_mem Buffer, pi_bool BlockingWrite,
    pi_buff_rect_offset BufferOffset, pi_buff_rect_offset HostOffset,
    pi_buff_rect_region Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, const void *Ptr,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *Event) {

  PI_ASSERT(Buffer, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  char *ZeHandleDst;
  PI_CALL(Buffer->getZeHandle(ZeHandleDst, _pi_mem::write_only, Queue->Device));
  return enqueueMemCopyRectHelper(
      PI_COMMAND_TYPE_MEM_BUFFER_WRITE_RECT, Queue,
      const_cast<char *>(static_cast<const char *>(Ptr)), ZeHandleDst,
      HostOffset, BufferOffset, Region, HostRowPitch, BufferRowPitch,
      HostSlicePitch, BufferSlicePitch, BlockingWrite, NumEventsInWaitList,
      EventWaitList, Event);
}

pi_result piEnqueueMemBufferCopy(pi_queue Queue, pi_mem SrcMem, pi_mem DstMem,
                                 size_t SrcOffset, size_t DstOffset,
                                 size_t Size, pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {
  PI_ASSERT(SrcMem && DstMem, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  PI_ASSERT(!SrcMem->isImage(), PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(!DstMem->isImage(), PI_ERROR_INVALID_MEM_OBJECT);
  auto SrcBuffer = ur_cast<pi_buffer>(SrcMem);
  auto DstBuffer = ur_cast<pi_buffer>(DstMem);

  std::shared_lock<ur_shared_mutex> SrcLock(SrcBuffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex,
                   ur_shared_mutex>
      LockAll(SrcLock, DstBuffer->Mutex, Queue->Mutex);

  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = (SrcBuffer->OnHost || DstBuffer->OnHost);

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  char *ZeHandleSrc;
  PI_CALL(
      SrcBuffer->getZeHandle(ZeHandleSrc, _pi_mem::read_only, Queue->Device));
  char *ZeHandleDst;
  PI_CALL(
      DstBuffer->getZeHandle(ZeHandleDst, _pi_mem::write_only, Queue->Device));

  return enqueueMemCopyHelper(
      PI_COMMAND_TYPE_MEM_BUFFER_COPY, Queue, ZeHandleDst + DstOffset,
      false, // blocking
      Size, ZeHandleSrc + SrcOffset, NumEventsInWaitList, EventWaitList, Event,
      PreferCopyEngine);
}

pi_result piEnqueueMemBufferCopyRect(
    pi_queue Queue, pi_mem SrcMem, pi_mem DstMem, pi_buff_rect_offset SrcOrigin,
    pi_buff_rect_offset DstOrigin, pi_buff_rect_region Region,
    size_t SrcRowPitch, size_t SrcSlicePitch, size_t DstRowPitch,
    size_t DstSlicePitch, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {
  PI_ASSERT(SrcMem && DstMem, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  PI_ASSERT(!SrcMem->isImage(), PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(!DstMem->isImage(), PI_ERROR_INVALID_MEM_OBJECT);
  auto SrcBuffer = ur_cast<pi_buffer>(SrcMem);
  auto DstBuffer = ur_cast<pi_buffer>(DstMem);

  std::shared_lock<ur_shared_mutex> SrcLock(SrcBuffer->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex,
                   ur_shared_mutex>
      LockAll(SrcLock, DstBuffer->Mutex, Queue->Mutex);

  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = (SrcBuffer->OnHost || DstBuffer->OnHost);

  char *ZeHandleSrc;
  PI_CALL(
      SrcBuffer->getZeHandle(ZeHandleSrc, _pi_mem::read_only, Queue->Device));
  char *ZeHandleDst;
  PI_CALL(
      DstBuffer->getZeHandle(ZeHandleDst, _pi_mem::write_only, Queue->Device));

  return enqueueMemCopyRectHelper(
      PI_COMMAND_TYPE_MEM_BUFFER_COPY_RECT, Queue, ZeHandleSrc, ZeHandleDst,
      SrcOrigin, DstOrigin, Region, SrcRowPitch, DstRowPitch, SrcSlicePitch,
      DstSlicePitch,
      false, // blocking
      NumEventsInWaitList, EventWaitList, Event, PreferCopyEngine);
}

} // extern "C"

// Default to using compute engine for fill operation, but allow to
// override this with an environment variable.
static bool PreferCopyEngine = [] {
  const char *UrRet = std::getenv("UR_L0_USE_COPY_ENGINE_FOR_FILL");
  const char *PiRet =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_FILL");
  return (UrRet ? std::stoi(UrRet) : (PiRet ? std::stoi(PiRet) : 0));
}();

// PI interfaces must have queue's and buffer's mutexes locked on entry.
static pi_result
enqueueMemFillHelper(pi_command_type CommandType, pi_queue Queue, void *Ptr,
                     const void *Pattern, size_t PatternSize, size_t Size,
                     pi_uint32 NumEventsInWaitList,
                     const pi_event *EventWaitList, pi_event *OutEvent) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);
  // Pattern size must be a power of two.
  PI_ASSERT((PatternSize > 0) && ((PatternSize & (PatternSize - 1)) == 0),
            PI_ERROR_INVALID_VALUE);

  auto &Device = Queue->Device;

  // Make sure that pattern size matches the capability of the copy queues.
  // Check both main and link groups as we don't known which one will be used.
  //
  if (PreferCopyEngine && Device->hasCopyEngine()) {
    if (Device->hasMainCopyEngine() &&
        Device->QueueGroup[_pi_device::queue_group_info_t::MainCopy]
                .ZeProperties.maxMemoryFillPatternSize < PatternSize) {
      PreferCopyEngine = false;
    }
    if (Device->hasLinkCopyEngine() &&
        Device->QueueGroup[_pi_device::queue_group_info_t::LinkCopy]
                .ZeProperties.maxMemoryFillPatternSize < PatternSize) {
      PreferCopyEngine = false;
    }
  }

  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);
  if (!UseCopyEngine) {
    // Pattern size must fit the compute queue capabilities.
    PI_ASSERT(PatternSize <=
                  Device->QueueGroup[_pi_device::queue_group_info_t::Compute]
                      .ZeProperties.maxMemoryFillPatternSize,
              PI_ERROR_INVALID_VALUE);
  }

  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
          NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine))
    return Res;

  pi_command_list_ptr_t CommandList{};
  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, CommandList, UseCopyEngine, OkToBatch))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  pi_event InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  pi_event *Event = OutEvent ? OutEvent : &InternalEvent;
  auto Res = createEventAndAssociateQueue(Queue, Event, CommandType,
                                          CommandList, IsInternal);
  if (Res != PI_SUCCESS)
    return Res;

  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  ZE_CALL(zeCommandListAppendMemoryFill,
          (ZeCommandList, Ptr, Pattern, PatternSize, Size, ZeEvent,
           WaitList.Length, WaitList.ZeEventList));

  urPrint("calling zeCommandListAppendMemoryFill() with\n"
          "  ZeEvent %#llx\n",
          ur_cast<pi_uint64>(ZeEvent));
  printZeEventList(WaitList);

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  if (auto Res = Queue->executeCommandList(CommandList, false, OkToBatch))
    return Res;

  return PI_SUCCESS;
}

extern "C" {

pi_result piEnqueueMemBufferFill(pi_queue Queue, pi_mem Buffer,
                                 const void *Pattern, size_t PatternSize,
                                 size_t Offset, size_t Size,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {

  PI_ASSERT(Buffer, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  char *ZeHandleDst;
  PI_CALL(Buffer->getZeHandle(ZeHandleDst, _pi_mem::write_only, Queue->Device));
  return enqueueMemFillHelper(PI_COMMAND_TYPE_MEM_BUFFER_FILL, Queue,
                              ZeHandleDst + Offset, Pattern, PatternSize, Size,
                              NumEventsInWaitList, EventWaitList, Event);
}

static pi_result USMHostAllocImpl(void **ResultPtr, pi_context Context,
                                  pi_usm_mem_properties *Properties,
                                  size_t Size, pi_uint32 Alignment);

pi_result piEnqueueMemBufferMap(pi_queue Queue, pi_mem Mem, pi_bool BlockingMap,
                                pi_map_flags MapFlags, size_t Offset,
                                size_t Size, pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *OutEvent, void **RetMap) {

  // TODO: we don't implement read-only or write-only, always read-write.
  // assert((map_flags & PI_MAP_READ) != 0);
  // assert((map_flags & PI_MAP_WRITE) != 0);
  PI_ASSERT(Mem, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  PI_ASSERT(!Mem->isImage(), PI_ERROR_INVALID_MEM_OBJECT);
  auto Buffer = ur_cast<pi_buffer>(Mem);

  pi_event InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  pi_event *Event = OutEvent ? OutEvent : &InternalEvent;
  ze_event_handle_t ZeEvent = nullptr;

  bool UseCopyEngine = false;
  {
    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

    _pi_ze_event_list_t TmpWaitList;
    if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
            NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine))
      return Res;

    auto Res = createEventAndAssociateQueue(
        Queue, Event, PI_COMMAND_TYPE_MEM_BUFFER_MAP,
        Queue->CommandListMap.end(), IsInternal);
    if (Res != PI_SUCCESS)
      return Res;

    ZeEvent = (*Event)->ZeEvent;
    (*Event)->WaitList = TmpWaitList;
  }

  // Translate the host access mode info.
  _pi_mem::access_mode_t AccessMode = _pi_mem::unknown;
  if (MapFlags & PI_MAP_WRITE_INVALIDATE_REGION)
    AccessMode = _pi_mem::write_only;
  else {
    if (MapFlags & PI_MAP_READ) {
      AccessMode = _pi_mem::read_only;
      if (MapFlags & PI_MAP_WRITE)
        AccessMode = _pi_mem::read_write;
    } else if (MapFlags & PI_MAP_WRITE)
      AccessMode = _pi_mem::write_only;
  }
  PI_ASSERT(AccessMode != _pi_mem::unknown, PI_ERROR_INVALID_VALUE);

  // TODO: Level Zero is missing the memory "mapping" capabilities, so we are
  // left to doing new memory allocation and a copy (read) on discrete devices.
  // For integrated devices, we have allocated the buffer in host memory so no
  // actions are needed here except for synchronizing on incoming events.
  // A host-to-host copy is done if a host pointer had been supplied during
  // buffer creation on integrated devices.
  //
  // TODO: for discrete, check if the input buffer is already allocated
  // in shared memory and thus is accessible from the host as is.
  // Can we get SYCL RT to predict/allocate in shared memory
  // from the beginning?

  // For integrated devices the buffer has been allocated in host memory.
  if (Buffer->OnHost) {
    // Wait on incoming events before doing the copy
    if (NumEventsInWaitList > 0)
      PI_CALL(piEventsWait(NumEventsInWaitList, EventWaitList));

    if (Queue->isInOrderQueue())
      PI_CALL(piQueueFinish(Queue));

    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> Guard(Buffer->Mutex);

    char *ZeHandleSrc;
    PI_CALL(Buffer->getZeHandle(ZeHandleSrc, AccessMode, Queue->Device));

    if (Buffer->MapHostPtr) {
      *RetMap = Buffer->MapHostPtr + Offset;
      if (ZeHandleSrc != Buffer->MapHostPtr &&
          AccessMode != _pi_mem::write_only) {
        memcpy(*RetMap, ZeHandleSrc + Offset, Size);
      }
    } else {
      *RetMap = ZeHandleSrc + Offset;
    }

    auto Res = Buffer->Mappings.insert({*RetMap, {Offset, Size}});
    // False as the second value in pair means that mapping was not inserted
    // because mapping already exists.
    if (!Res.second) {
      urPrint("piEnqueueMemBufferMap: duplicate mapping detected\n");
      return PI_ERROR_INVALID_VALUE;
    }

    // Signal this event
    ZE_CALL(zeEventHostSignal, (ZeEvent));
    (*Event)->Completed = true;
    return PI_SUCCESS;
  }

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  if (Buffer->MapHostPtr) {
    *RetMap = Buffer->MapHostPtr + Offset;
  } else {
    // TODO: use USM host allocator here
    // TODO: Do we even need every map to allocate new host memory?
    //       In the case when the buffer is "OnHost" we use single allocation.
    if (auto Res = ZeHostMemAllocHelper(RetMap, Queue->Context, Size))
      return Res;
  }

  // Take a shortcut if the host is not going to read buffer's data.
  if (AccessMode == _pi_mem::write_only) {
    (*Event)->Completed = true;
  } else {
    // For discrete devices we need a command list
    pi_command_list_ptr_t CommandList{};
    if (auto Res = Queue->Context->getAvailableCommandList(Queue, CommandList,
                                                           UseCopyEngine))
      return Res;

    // Add the event to the command list.
    CommandList->second.append(*Event);
    (*Event)->RefCount.increment();

    const auto &ZeCommandList = CommandList->first;
    const auto &WaitList = (*Event)->WaitList;

    char *ZeHandleSrc;
    PI_CALL(Buffer->getZeHandle(ZeHandleSrc, AccessMode, Queue->Device));

    ZE_CALL(zeCommandListAppendMemoryCopy,
            (ZeCommandList, *RetMap, ZeHandleSrc + Offset, Size, ZeEvent,
             WaitList.Length, WaitList.ZeEventList));

    if (auto Res = Queue->executeCommandList(CommandList, BlockingMap))
      return Res;
  }

  auto Res = Buffer->Mappings.insert({*RetMap, {Offset, Size}});
  // False as the second value in pair means that mapping was not inserted
  // because mapping already exists.
  if (!Res.second) {
    urPrint("piEnqueueMemBufferMap: duplicate mapping detected\n");
    return PI_ERROR_INVALID_VALUE;
  }
  return PI_SUCCESS;
}

pi_result piEnqueueMemUnmap(pi_queue Queue, pi_mem Mem, void *MappedPtr,
                            pi_uint32 NumEventsInWaitList,
                            const pi_event *EventWaitList, pi_event *OutEvent) {
  PI_ASSERT(Mem, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  PI_ASSERT(!Mem->isImage(), PI_ERROR_INVALID_MEM_OBJECT);
  auto Buffer = ur_cast<pi_buffer>(Mem);

  bool UseCopyEngine = false;

  ze_event_handle_t ZeEvent = nullptr;
  pi_event InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  pi_event *Event = OutEvent ? OutEvent : &InternalEvent;
  {
    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

    _pi_ze_event_list_t TmpWaitList;
    if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
            NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine))
      return Res;

    auto Res = createEventAndAssociateQueue(
        Queue, Event, PI_COMMAND_TYPE_MEM_BUFFER_UNMAP,
        Queue->CommandListMap.end(), IsInternal);
    if (Res != PI_SUCCESS)
      return Res;
    ZeEvent = (*Event)->ZeEvent;
    (*Event)->WaitList = TmpWaitList;
  }

  _pi_buffer::Mapping MapInfo = {};
  {
    // Lock automatically releases when this goes out of scope.
    std::scoped_lock<ur_shared_mutex> Guard(Buffer->Mutex);
    auto It = Buffer->Mappings.find(MappedPtr);
    if (It == Buffer->Mappings.end()) {
      urPrint("piEnqueueMemUnmap: unknown memory mapping\n");
      return PI_ERROR_INVALID_VALUE;
    }
    MapInfo = It->second;
    Buffer->Mappings.erase(It);

    // NOTE: we still have to free the host memory allocated/returned by
    // piEnqueueMemBufferMap, but can only do so after the above copy
    // is completed. Instead of waiting for It here (blocking), we shall
    // do so in piEventRelease called for the pi_event tracking the unmap.
    // In the case of an integrated device, the map operation does not allocate
    // any memory, so there is nothing to free. This is indicated by a nullptr.
    (*Event)->CommandData =
        (Buffer->OnHost ? nullptr : (Buffer->MapHostPtr ? nullptr : MappedPtr));
  }

  // For integrated devices the buffer is allocated in host memory.
  if (Buffer->OnHost) {
    // Wait on incoming events before doing the copy
    if (NumEventsInWaitList > 0)
      PI_CALL(piEventsWait(NumEventsInWaitList, EventWaitList));

    if (Queue->isInOrderQueue())
      PI_CALL(piQueueFinish(Queue));

    char *ZeHandleDst;
    PI_CALL(
        Buffer->getZeHandle(ZeHandleDst, _pi_mem::write_only, Queue->Device));

    std::scoped_lock<ur_shared_mutex> Guard(Buffer->Mutex);
    if (Buffer->MapHostPtr)
      memcpy(ZeHandleDst + MapInfo.Offset, MappedPtr, MapInfo.Size);

    // Signal this event
    ZE_CALL(zeEventHostSignal, (ZeEvent));
    (*Event)->Completed = true;
    return PI_SUCCESS;
  }

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Buffer->Mutex);

  pi_command_list_ptr_t CommandList{};
  if (auto Res = Queue->Context->getAvailableCommandList(Queue, CommandList,
                                                         UseCopyEngine))
    return Res;

  CommandList->second.append(*Event);
  (*Event)->RefCount.increment();

  const auto &ZeCommandList = CommandList->first;

  // TODO: Level Zero is missing the memory "mapping" capabilities, so we are
  // left to doing copy (write back to the device).
  //
  // NOTE: Keep this in sync with the implementation of
  // piEnqueueMemBufferMap.

  char *ZeHandleDst;
  PI_CALL(Buffer->getZeHandle(ZeHandleDst, _pi_mem::write_only, Queue->Device));

  ZE_CALL(zeCommandListAppendMemoryCopy,
          (ZeCommandList, ZeHandleDst + MapInfo.Offset, MappedPtr, MapInfo.Size,
           ZeEvent, (*Event)->WaitList.Length, (*Event)->WaitList.ZeEventList));

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  if (auto Res = Queue->executeCommandList(CommandList))
    return Res;

  return PI_SUCCESS;
}

pi_result piMemImageGetInfo(pi_mem Image, pi_image_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {
  (void)Image;
  (void)ParamName;
  (void)ParamValueSize;
  (void)ParamValue;
  (void)ParamValueSizeRet;

  die("piMemImageGetInfo: not implemented");
  return {};
}

} // extern "C"

static pi_result getImageRegionHelper(pi_mem Mem, pi_image_offset Origin,
                                      pi_image_region Region,
                                      ze_image_region_t &ZeRegion) {

  PI_ASSERT(Mem, PI_ERROR_INVALID_MEM_OBJECT);
  PI_ASSERT(Origin, PI_ERROR_INVALID_VALUE);

#ifndef NDEBUG
  PI_ASSERT(Mem->isImage(), PI_ERROR_INVALID_MEM_OBJECT);
  auto Image = static_cast<_pi_image *>(Mem);
  ze_image_desc_t &ZeImageDesc = Image->ZeImageDesc;

  PI_ASSERT((ZeImageDesc.type == ZE_IMAGE_TYPE_1D && Origin->y == 0 &&
             Origin->z == 0) ||
                (ZeImageDesc.type == ZE_IMAGE_TYPE_1DARRAY && Origin->z == 0) ||
                (ZeImageDesc.type == ZE_IMAGE_TYPE_2D && Origin->z == 0) ||
                (ZeImageDesc.type == ZE_IMAGE_TYPE_3D),
            PI_ERROR_INVALID_VALUE);

  PI_ASSERT(Region->width && Region->height && Region->depth,
            PI_ERROR_INVALID_VALUE);
  PI_ASSERT(
      (ZeImageDesc.type == ZE_IMAGE_TYPE_1D && Region->height == 1 &&
       Region->depth == 1) ||
          (ZeImageDesc.type == ZE_IMAGE_TYPE_1DARRAY && Region->depth == 1) ||
          (ZeImageDesc.type == ZE_IMAGE_TYPE_2D && Region->depth == 1) ||
          (ZeImageDesc.type == ZE_IMAGE_TYPE_3D),
      PI_ERROR_INVALID_VALUE);
#endif // !NDEBUG

  uint32_t OriginX = ur_cast<uint32_t>(Origin->x);
  uint32_t OriginY = ur_cast<uint32_t>(Origin->y);
  uint32_t OriginZ = ur_cast<uint32_t>(Origin->z);

  uint32_t Width = ur_cast<uint32_t>(Region->width);
  uint32_t Height = ur_cast<uint32_t>(Region->height);
  uint32_t Depth = ur_cast<uint32_t>(Region->depth);

  ZeRegion = {OriginX, OriginY, OriginZ, Width, Height, Depth};

  return PI_SUCCESS;
}

// Helper function to implement image read/write/copy.
// PI interfaces must have queue's and destination image's mutexes locked for
// exclusive use and source image's mutex locked for shared use on entry.
static pi_result enqueueMemImageCommandHelper(
    pi_command_type CommandType, pi_queue Queue,
    const void *Src, // image or ptr
    void *Dst,       // image or ptr
    pi_bool IsBlocking, pi_image_offset SrcOrigin, pi_image_offset DstOrigin,
    pi_image_region Region, size_t RowPitch, size_t SlicePitch,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *OutEvent, bool PreferCopyEngine = false) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);

  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
          NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine))
    return Res;

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  // Get a new command list to be used on this call
  pi_command_list_ptr_t CommandList{};
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, CommandList, UseCopyEngine, OkToBatch))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  pi_event InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  pi_event *Event = OutEvent ? OutEvent : &InternalEvent;
  auto Res = createEventAndAssociateQueue(Queue, Event, CommandType,
                                          CommandList, IsInternal);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  if (CommandType == PI_COMMAND_TYPE_IMAGE_READ) {
    pi_mem SrcMem = ur_cast<pi_mem>(const_cast<void *>(Src));

    ze_image_region_t ZeSrcRegion;
    auto Result = getImageRegionHelper(SrcMem, SrcOrigin, Region, ZeSrcRegion);
    if (Result != PI_SUCCESS)
      return Result;

    // TODO: Level Zero does not support row_pitch/slice_pitch for images yet.
    // Check that SYCL RT did not want pitch larger than default.
    (void)RowPitch;
    (void)SlicePitch;
#ifndef NDEBUG
    PI_ASSERT(SrcMem->isImage(), PI_ERROR_INVALID_MEM_OBJECT);

    auto SrcImage = static_cast<_pi_image *>(SrcMem);
    const ze_image_desc_t &ZeImageDesc = SrcImage->ZeImageDesc;
    PI_ASSERT(
        RowPitch == 0 ||
            // special case RGBA image pitch equal to region's width
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 &&
             RowPitch == 4 * 4 * ZeSrcRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 &&
             RowPitch == 4 * 2 * ZeSrcRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 &&
             RowPitch == 4 * ZeSrcRegion.width),
        PI_ERROR_INVALID_IMAGE_SIZE);
    PI_ASSERT(SlicePitch == 0 || SlicePitch == RowPitch * ZeSrcRegion.height,
              PI_ERROR_INVALID_IMAGE_SIZE);
#endif // !NDEBUG

    char *ZeHandleSrc;
    PI_CALL(
        SrcMem->getZeHandle(ZeHandleSrc, _pi_mem::read_only, Queue->Device));
    ZE_CALL(zeCommandListAppendImageCopyToMemory,
            (ZeCommandList, Dst, ur_cast<ze_image_handle_t>(ZeHandleSrc),
             &ZeSrcRegion, ZeEvent, WaitList.Length, WaitList.ZeEventList));
  } else if (CommandType == PI_COMMAND_TYPE_IMAGE_WRITE) {
    pi_mem DstMem = ur_cast<pi_mem>(Dst);
    ze_image_region_t ZeDstRegion;
    auto Result = getImageRegionHelper(DstMem, DstOrigin, Region, ZeDstRegion);
    if (Result != PI_SUCCESS)
      return Result;

      // TODO: Level Zero does not support row_pitch/slice_pitch for images yet.
      // Check that SYCL RT did not want pitch larger than default.
#ifndef NDEBUG
    PI_ASSERT(DstMem->isImage(), PI_ERROR_INVALID_MEM_OBJECT);

    auto DstImage = static_cast<_pi_image *>(DstMem);
    const ze_image_desc_t &ZeImageDesc = DstImage->ZeImageDesc;
    PI_ASSERT(
        RowPitch == 0 ||
            // special case RGBA image pitch equal to region's width
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 &&
             RowPitch == 4 * 4 * ZeDstRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 &&
             RowPitch == 4 * 2 * ZeDstRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 &&
             RowPitch == 4 * ZeDstRegion.width),
        PI_ERROR_INVALID_IMAGE_SIZE);
    PI_ASSERT(SlicePitch == 0 || SlicePitch == RowPitch * ZeDstRegion.height,
              PI_ERROR_INVALID_IMAGE_SIZE);
#endif // !NDEBUG

    char *ZeHandleDst;
    PI_CALL(
        DstMem->getZeHandle(ZeHandleDst, _pi_mem::write_only, Queue->Device));
    ZE_CALL(zeCommandListAppendImageCopyFromMemory,
            (ZeCommandList, ur_cast<ze_image_handle_t>(ZeHandleDst), Src,
             &ZeDstRegion, ZeEvent, WaitList.Length, WaitList.ZeEventList));
  } else if (CommandType == PI_COMMAND_TYPE_IMAGE_COPY) {
    pi_mem SrcImage = ur_cast<pi_mem>(const_cast<void *>(Src));
    pi_mem DstImage = ur_cast<pi_mem>(Dst);

    ze_image_region_t ZeSrcRegion;
    auto Result =
        getImageRegionHelper(SrcImage, SrcOrigin, Region, ZeSrcRegion);
    if (Result != PI_SUCCESS)
      return Result;
    ze_image_region_t ZeDstRegion;
    Result = getImageRegionHelper(DstImage, DstOrigin, Region, ZeDstRegion);
    if (Result != PI_SUCCESS)
      return Result;

    char *ZeHandleSrc;
    char *ZeHandleDst;
    PI_CALL(
        SrcImage->getZeHandle(ZeHandleSrc, _pi_mem::read_only, Queue->Device));
    PI_CALL(
        DstImage->getZeHandle(ZeHandleDst, _pi_mem::write_only, Queue->Device));
    ZE_CALL(zeCommandListAppendImageCopyRegion,
            (ZeCommandList, ur_cast<ze_image_handle_t>(ZeHandleDst),
             ur_cast<ze_image_handle_t>(ZeHandleSrc), &ZeDstRegion,
             &ZeSrcRegion, ZeEvent, 0, nullptr));
  } else {
    urPrint("enqueueMemImageUpdate: unsupported image command type\n");
    return PI_ERROR_INVALID_OPERATION;
  }

  if (auto Res = Queue->executeCommandList(CommandList, IsBlocking, OkToBatch))
    return Res;

  return PI_SUCCESS;
}

extern "C" {

pi_result piEnqueueMemImageRead(pi_queue Queue, pi_mem Image,
                                pi_bool BlockingRead, pi_image_offset Origin,
                                pi_image_region Region, size_t RowPitch,
                                size_t SlicePitch, void *Ptr,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  std::shared_lock<ur_shared_mutex> SrcLock(Image->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex> LockAll(
      SrcLock, Queue->Mutex);
  return enqueueMemImageCommandHelper(
      PI_COMMAND_TYPE_IMAGE_READ, Queue,
      Image, // src
      Ptr,   // dst
      BlockingRead,
      Origin,  // SrcOrigin
      nullptr, // DstOrigin
      Region, RowPitch, SlicePitch, NumEventsInWaitList, EventWaitList, Event);
}

pi_result piEnqueueMemImageWrite(pi_queue Queue, pi_mem Image,
                                 pi_bool BlockingWrite, pi_image_offset Origin,
                                 pi_image_region Region, size_t InputRowPitch,
                                 size_t InputSlicePitch, const void *Ptr,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Image->Mutex);
  return enqueueMemImageCommandHelper(PI_COMMAND_TYPE_IMAGE_WRITE, Queue,
                                      Ptr,   // src
                                      Image, // dst
                                      BlockingWrite,
                                      nullptr, // SrcOrigin
                                      Origin,  // DstOrigin
                                      Region, InputRowPitch, InputSlicePitch,
                                      NumEventsInWaitList, EventWaitList,
                                      Event);
}

pi_result
piEnqueueMemImageCopy(pi_queue Queue, pi_mem SrcImage, pi_mem DstImage,
                      pi_image_offset SrcOrigin, pi_image_offset DstOrigin,
                      pi_image_region Region, pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *Event) {

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  std::shared_lock<ur_shared_mutex> SrcLock(SrcImage->Mutex, std::defer_lock);
  std::scoped_lock<std::shared_lock<ur_shared_mutex>, ur_shared_mutex,
                   ur_shared_mutex>
      LockAll(SrcLock, DstImage->Mutex, Queue->Mutex);
  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  // Images are always allocated on device.
  bool PreferCopyEngine = false;
  return enqueueMemImageCommandHelper(
      PI_COMMAND_TYPE_IMAGE_COPY, Queue, SrcImage, DstImage,
      false, // is_blocking
      SrcOrigin, DstOrigin, Region,
      0, // row pitch
      0, // slice pitch
      NumEventsInWaitList, EventWaitList, Event, PreferCopyEngine);
}

pi_result piEnqueueMemImageFill(pi_queue Queue, pi_mem Image,
                                const void *FillColor, const size_t *Origin,
                                const size_t *Region,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {
  (void)Image;
  (void)FillColor;
  (void)Origin;
  (void)Region;
  (void)NumEventsInWaitList;
  (void)EventWaitList;
  (void)Event;

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex, ur_shared_mutex> Lock(Queue->Mutex,
                                                          Image->Mutex);

  die("piEnqueueMemImageFill: not implemented");
  return {};
}

pi_result piMemBufferPartition(pi_mem Buffer, pi_mem_flags Flags,
                               pi_buffer_create_type BufferCreateType,
                               void *BufferCreateInfo, pi_mem *RetMem) {

  PI_ASSERT(Buffer && !Buffer->isImage() &&
                !(static_cast<pi_buffer>(Buffer))->isSubBuffer(),
            PI_ERROR_INVALID_MEM_OBJECT);

  PI_ASSERT(BufferCreateType == PI_BUFFER_CREATE_TYPE_REGION &&
                BufferCreateInfo && RetMem,
            PI_ERROR_INVALID_VALUE);

  std::shared_lock<ur_shared_mutex> Guard(Buffer->Mutex);

  if (Flags != PI_MEM_FLAGS_ACCESS_RW) {
    die("piMemBufferPartition: Level-Zero implements only read-write buffer,"
        "no read-only or write-only yet.");
  }

  auto Region = (pi_buffer_region)BufferCreateInfo;

  PI_ASSERT(Region->size != 0u, PI_ERROR_INVALID_BUFFER_SIZE);
  PI_ASSERT(Region->origin <= (Region->origin + Region->size),
            PI_ERROR_INVALID_VALUE);

  try {
    *RetMem = new _pi_buffer(static_cast<pi_buffer>(Buffer), Region->origin,
                             Region->size);
  } catch (const std::bad_alloc &) {
    return PI_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piEnqueueNativeKernel(pi_queue Queue, void (*UserFunc)(void *),
                                void *Args, size_t CbArgs,
                                pi_uint32 NumMemObjects, const pi_mem *MemList,
                                const void **ArgsMemLoc,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {
  (void)UserFunc;
  (void)Args;
  (void)CbArgs;
  (void)NumMemObjects;
  (void)MemList;
  (void)ArgsMemLoc;
  (void)NumEventsInWaitList;
  (void)EventWaitList;
  (void)Event;

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  die("piEnqueueNativeKernel: not implemented");
  return {};
}

// Function gets characters between delimeter's in str
// then checks if they are equal to the sub_str.
// returns true if there is at least one instance
// returns false if there are no instances of the name
static bool is_in_separated_string(const std::string &str, char delimiter,
                                   const std::string &sub_str) {
  size_t beg = 0;
  size_t length = 0;
  for (const auto &x : str) {
    if (x == delimiter) {
      if (str.substr(beg, length) == sub_str)
        return true;

      beg += length + 1;
      length = 0;
      continue;
    }
    length++;
  }
  if (length != 0)
    if (str.substr(beg, length) == sub_str)
      return true;

  return false;
}

// TODO: Check if the function_pointer_ret type can be converted to void**.
pi_result piextGetDeviceFunctionPointer(pi_device Device, pi_program Program,
                                        const char *FunctionName,
                                        pi_uint64 *FunctionPointerRet) {
  (void)Device;
  PI_ASSERT(Program, PI_ERROR_INVALID_PROGRAM);

  std::shared_lock<ur_shared_mutex> Guard(Program->Mutex);
  if (Program->State != _pi_program::Exe) {
    return PI_ERROR_INVALID_PROGRAM_EXECUTABLE;
  }

  ze_result_t ZeResult =
      ZE_CALL_NOCHECK(zeModuleGetFunctionPointer,
                      (Program->ZeModule, FunctionName,
                       reinterpret_cast<void **>(FunctionPointerRet)));

  // zeModuleGetFunctionPointer currently fails for all
  // kernels regardless of if the kernel exist or not
  // with ZE_RESULT_ERROR_INVALID_ARGUMENT
  // TODO: remove when this is no longer the case
  // If zeModuleGetFunctionPointer returns invalid argument,
  // fallback to searching through kernel list and return
  // PI_ERROR_FUNCTION_ADDRESS_IS_NOT_AVAILABLE if the function exists
  // or PI_ERROR_INVALID_KERNEL_NAME if the function does not exist.
  // FunctionPointerRet should always be 0
  if (ZeResult == ZE_RESULT_ERROR_INVALID_ARGUMENT) {
    size_t Size;
    *FunctionPointerRet = 0;
    PI_CALL(piProgramGetInfo(Program, PI_PROGRAM_INFO_KERNEL_NAMES, 0, nullptr,
                             &Size));

    std::string ClResult(Size, ' ');
    PI_CALL(piProgramGetInfo(Program, PI_PROGRAM_INFO_KERNEL_NAMES,
                             ClResult.size(), &ClResult[0], nullptr));

    // Get rid of the null terminator and search for kernel_name
    // If function can be found return error code to indicate it
    // exists
    ClResult.pop_back();
    if (is_in_separated_string(ClResult, ';', std::string(FunctionName)))
      return PI_ERROR_FUNCTION_ADDRESS_IS_NOT_AVAILABLE;

    return PI_ERROR_INVALID_KERNEL_NAME;
  }

  if (ZeResult == ZE_RESULT_ERROR_INVALID_FUNCTION_NAME) {
    *FunctionPointerRet = 0;
    return PI_ERROR_INVALID_KERNEL_NAME;
  }

  return mapError(ZeResult);
}

enum class USMAllocationForceResidencyType {
  // Do not force memory residency at allocation time.
  None = 0,
  // Force memory resident on the device of allocation at allocation time.
  // For host allocation force residency on all devices in a context.
  Device = 1,
  // Force memory resident on all devices in the context with P2P
  // access to the device of allocation.
  // For host allocation force residency on all devices in a context.
  P2PDevices = 2
};

// Returns the desired USM residency setting
// Input value is of the form 0xHSD, where:
//   4-bits of D control device allocations
//   4-bits of S control shared allocations
//   4-bits of H control host allocations
// Each 4-bit value is holding a USMAllocationForceResidencyType enum value.
// The default is 0x2, i.e. force full residency for device allocations only.
//
static uint32_t USMAllocationForceResidency = [] {
  const char *UrRet = std::getenv("UR_L0_USM_RESIDENT");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_USM_RESIDENT");
  const char *Str = UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  try {
    if (Str) {
      // Auto-detect radix to allow more convinient hex base
      return std::stoi(Str, nullptr, 0);
    }
  } catch (...) {
  }
  return 0x2;
}();

// Convert from an integer value to USMAllocationForceResidencyType enum value
static USMAllocationForceResidencyType
USMAllocationForceResidencyConvert(uint32_t Val) {
  switch (Val) {
  case 1:
    return USMAllocationForceResidencyType::Device;
  case 2:
    return USMAllocationForceResidencyType::P2PDevices;
  default:
    return USMAllocationForceResidencyType::None;
  };
}

static USMAllocationForceResidencyType USMHostAllocationForceResidency = [] {
  return USMAllocationForceResidencyConvert(
      (USMAllocationForceResidency & 0xf00) >> 8);
}();
static USMAllocationForceResidencyType USMSharedAllocationForceResidency = [] {
  return USMAllocationForceResidencyConvert(
      (USMAllocationForceResidency & 0x0f0) >> 4);
}();
static USMAllocationForceResidencyType USMDeviceAllocationForceResidency = [] {
  return USMAllocationForceResidencyConvert(
      (USMAllocationForceResidency & 0x00f));
}();

// Make USM allocation resident as requested
static pi_result
USMAllocationMakeResident(USMAllocationForceResidencyType ForceResidency,
                          pi_context Context,
                          pi_device Device, // nullptr for host allocation
                          void *Ptr, size_t Size) {
  if (ForceResidency == USMAllocationForceResidencyType::None)
    return PI_SUCCESS;

  std::list<pi_device> Devices;
  if (!Device) {
    // Host allocation, make it resident on all devices in the context
    Devices.insert(Devices.end(), Context->Devices.begin(),
                   Context->Devices.end());
  } else {
    Devices.push_back(Device);
    if (ForceResidency == USMAllocationForceResidencyType::P2PDevices) {
      ze_bool_t P2P;
      for (const auto &D : Context->Devices) {
        if (D == Device)
          continue;
        // TODO: Cache P2P devices for a context
        ZE_CALL(zeDeviceCanAccessPeer, (D->ZeDevice, Device->ZeDevice, &P2P));
        if (P2P)
          Devices.push_back(D);
      }
    }
  }
  for (const auto &D : Devices) {
    ZE_CALL(zeContextMakeMemoryResident,
            (Context->ZeContext, D->ZeDevice, Ptr, Size));
  }
  return PI_SUCCESS;
}

static pi_result USMDeviceAllocImpl(void **ResultPtr, pi_context Context,
                                    pi_device Device,
                                    pi_usm_mem_properties *Properties,
                                    size_t Size, pi_uint32 Alignment) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  // Check that incorrect bits are not set in the properties.
  PI_ASSERT(!Properties || *Properties == 0 ||
                (*Properties == PI_MEM_ALLOC_FLAGS && *(Properties + 2) == 0),
            PI_ERROR_INVALID_VALUE);

  // TODO: translate PI properties to Level Zero flags
  ZeStruct<ze_device_mem_alloc_desc_t> ZeDesc;
  ZeDesc.flags = 0;
  ZeDesc.ordinal = 0;

  ZeStruct<ze_relaxed_allocation_limits_exp_desc_t> RelaxedDesc;
  if (Size > Device->ZeDeviceProperties->maxMemAllocSize) {
    // Tell Level-Zero to accept Size > maxMemAllocSize
    RelaxedDesc.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
    ZeDesc.pNext = &RelaxedDesc;
  }

  ZE_CALL(zeMemAllocDevice, (Context->ZeContext, &ZeDesc, Size, Alignment,
                             Device->ZeDevice, ResultPtr));

  PI_ASSERT(Alignment == 0 ||
                reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0,
            PI_ERROR_INVALID_VALUE);

  USMAllocationMakeResident(USMDeviceAllocationForceResidency, Context, Device,
                            *ResultPtr, Size);
  return PI_SUCCESS;
}

static pi_result USMSharedAllocImpl(void **ResultPtr, pi_context Context,
                                    pi_device Device, pi_usm_mem_properties *,
                                    size_t Size, pi_uint32 Alignment) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

  // TODO: translate PI properties to Level Zero flags
  ZeStruct<ze_host_mem_alloc_desc_t> ZeHostDesc;
  ZeHostDesc.flags = 0;
  ZeStruct<ze_device_mem_alloc_desc_t> ZeDevDesc;
  ZeDevDesc.flags = 0;
  ZeDevDesc.ordinal = 0;

  ZeStruct<ze_relaxed_allocation_limits_exp_desc_t> RelaxedDesc;
  if (Size > Device->ZeDeviceProperties->maxMemAllocSize) {
    // Tell Level-Zero to accept Size > maxMemAllocSize
    RelaxedDesc.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
    ZeDevDesc.pNext = &RelaxedDesc;
  }

  ZE_CALL(zeMemAllocShared, (Context->ZeContext, &ZeDevDesc, &ZeHostDesc, Size,
                             Alignment, Device->ZeDevice, ResultPtr));

  PI_ASSERT(Alignment == 0 ||
                reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0,
            PI_ERROR_INVALID_VALUE);

  USMAllocationMakeResident(USMSharedAllocationForceResidency, Context, Device,
                            *ResultPtr, Size);

  // TODO: Handle PI_MEM_ALLOC_DEVICE_READ_ONLY.
  return PI_SUCCESS;
}

static pi_result USMHostAllocImpl(void **ResultPtr, pi_context Context,
                                  pi_usm_mem_properties *Properties,
                                  size_t Size, pi_uint32 Alignment) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  // Check that incorrect bits are not set in the properties.
  PI_ASSERT(!Properties || *Properties == 0 ||
                (*Properties == PI_MEM_ALLOC_FLAGS && *(Properties + 2) == 0),
            PI_ERROR_INVALID_VALUE);

  // TODO: translate PI properties to Level Zero flags
  ZeStruct<ze_host_mem_alloc_desc_t> ZeHostDesc;
  ZeHostDesc.flags = 0;
  ZE_CALL(zeMemAllocHost,
          (Context->ZeContext, &ZeHostDesc, Size, Alignment, ResultPtr));

  PI_ASSERT(Alignment == 0 ||
                reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0,
            PI_ERROR_INVALID_VALUE);

  USMAllocationMakeResident(USMHostAllocationForceResidency, Context, nullptr,
                            *ResultPtr, Size);
  return PI_SUCCESS;
}

static pi_result USMFreeImpl(pi_context Context, void *Ptr) {
  ZE_CALL(zeMemFree, (Context->ZeContext, Ptr));
  return PI_SUCCESS;
}

// Exception type to pass allocation errors
class UsmAllocationException {
  const pi_result Error;

public:
  UsmAllocationException(pi_result Err) : Error{Err} {}
  pi_result getError() const { return Error; }
};

pi_result USMSharedMemoryAlloc::allocateImpl(void **ResultPtr, size_t Size,
                                             pi_uint32 Alignment) {
  return USMSharedAllocImpl(ResultPtr, Context, Device, nullptr, Size,
                            Alignment);
}

pi_result USMSharedReadOnlyMemoryAlloc::allocateImpl(void **ResultPtr,
                                                     size_t Size,
                                                     pi_uint32 Alignment) {
  pi_usm_mem_properties Props[] = {PI_MEM_ALLOC_FLAGS,
                                   PI_MEM_ALLOC_DEVICE_READ_ONLY, 0};
  return USMSharedAllocImpl(ResultPtr, Context, Device, Props, Size, Alignment);
}

pi_result USMDeviceMemoryAlloc::allocateImpl(void **ResultPtr, size_t Size,
                                             pi_uint32 Alignment) {
  return USMDeviceAllocImpl(ResultPtr, Context, Device, nullptr, Size,
                            Alignment);
}

pi_result USMHostMemoryAlloc::allocateImpl(void **ResultPtr, size_t Size,
                                           pi_uint32 Alignment) {
  return USMHostAllocImpl(ResultPtr, Context, nullptr, Size, Alignment);
}

void *USMMemoryAllocBase::allocate(size_t Size) {
  void *Ptr = nullptr;

  auto Res = allocateImpl(&Ptr, Size, sizeof(void *));
  if (Res != PI_SUCCESS) {
    throw UsmAllocationException(Res);
  }

  return Ptr;
}

void *USMMemoryAllocBase::allocate(size_t Size, size_t Alignment) {
  void *Ptr = nullptr;

  auto Res = allocateImpl(&Ptr, Size, Alignment);
  if (Res != PI_SUCCESS) {
    throw UsmAllocationException(Res);
  }
  return Ptr;
}

void USMMemoryAllocBase::deallocate(void *Ptr) {
  auto Res = USMFreeImpl(Context, Ptr);
  if (Res != PI_SUCCESS) {
    throw UsmAllocationException(Res);
  }
}

pi_result piextUSMDeviceAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {
  // L0 supports alignment up to 64KB and silently ignores higher values.
  // We flag alignment > 64KB as an invalid value.
  if (Alignment > 65536)
    return PI_ERROR_INVALID_VALUE;

  pi_platform Plt = Device->Platform;

  // If indirect access tracking is enabled then lock the mutex which is
  // guarding contexts container in the platform. This prevents new kernels from
  // being submitted in any context while we are in the process of allocating a
  // memory, this is needed to properly capture allocations by kernels with
  // indirect access. This lock also protects access to the context's data
  // structures. If indirect access tracking is not enabled then lock context
  // mutex to protect access to context's data structures.
  std::shared_lock<ur_shared_mutex> ContextLock(Context->Mutex,
                                                std::defer_lock);
  std::unique_lock<ur_shared_mutex> IndirectAccessTrackingLock(
      Plt->ContextsMutex, std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    IndirectAccessTrackingLock.lock();
    // We are going to defer memory release if there are kernels with indirect
    // access, that is why explicitly retain context to be sure that it is
    // released after all memory allocations in this context are released.
    PI_CALL(piContextRetain(Context));
  } else {
    ContextLock.lock();
  }

  if (!UseUSMAllocator ||
      // L0 spec says that allocation fails if Alignment != 2^n, in order to
      // keep the same behavior for the allocator, just call L0 API directly and
      // return the error code.
      ((Alignment & (Alignment - 1)) != 0)) {
    pi_result Res = USMDeviceAllocImpl(ResultPtr, Context, Device, Properties,
                                       Size, Alignment);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*ResultPtr),
                                 std::forward_as_tuple(Context));
    }
    return Res;
  }

  try {
    auto It = Context->DeviceMemAllocContexts.find(Device->ZeDevice);
    if (It == Context->DeviceMemAllocContexts.end())
      return PI_ERROR_INVALID_VALUE;

    *ResultPtr = It->second.allocate(Size, Alignment);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*ResultPtr),
                                 std::forward_as_tuple(Context));
    }

  } catch (const UsmAllocationException &Ex) {
    *ResultPtr = nullptr;
    return Ex.getError();
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piextUSMSharedAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {
  // See if the memory is going to be read-only on the device.
  bool DeviceReadOnly = false;
  // Check that incorrect bits are not set in the properties.
  if (Properties && *Properties != 0) {
    PI_ASSERT(*(Properties) == PI_MEM_ALLOC_FLAGS && *(Properties + 2) == 0,
              PI_ERROR_INVALID_VALUE);
    DeviceReadOnly = *(Properties + 1) & PI_MEM_ALLOC_DEVICE_READ_ONLY;
  }

  // L0 supports alignment up to 64KB and silently ignores higher values.
  // We flag alignment > 64KB as an invalid value.
  if (Alignment > 65536)
    return PI_ERROR_INVALID_VALUE;

  pi_platform Plt = Device->Platform;

  // If indirect access tracking is enabled then lock the mutex which is
  // guarding contexts container in the platform. This prevents new kernels from
  // being submitted in any context while we are in the process of allocating a
  // memory, this is needed to properly capture allocations by kernels with
  // indirect access. This lock also protects access to the context's data
  // structures. If indirect access tracking is not enabled then lock context
  // mutex to protect access to context's data structures.
  std::scoped_lock<ur_shared_mutex> Lock(
      IndirectAccessTrackingEnabled ? Plt->ContextsMutex : Context->Mutex);

  if (IndirectAccessTrackingEnabled) {
    // We are going to defer memory release if there are kernels with indirect
    // access, that is why explicitly retain context to be sure that it is
    // released after all memory allocations in this context are released.
    PI_CALL(piContextRetain(Context));
  }

  if (!UseUSMAllocator ||
      // L0 spec says that allocation fails if Alignment != 2^n, in order to
      // keep the same behavior for the allocator, just call L0 API directly and
      // return the error code.
      ((Alignment & (Alignment - 1)) != 0)) {
    pi_result Res = USMSharedAllocImpl(ResultPtr, Context, Device, Properties,
                                       Size, Alignment);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*ResultPtr),
                                 std::forward_as_tuple(Context));
    }
    return Res;
  }

  try {
    auto &Allocator = (DeviceReadOnly ? Context->SharedReadOnlyMemAllocContexts
                                      : Context->SharedMemAllocContexts);
    auto It = Allocator.find(Device->ZeDevice);
    if (It == Allocator.end())
      return PI_ERROR_INVALID_VALUE;

    *ResultPtr = It->second.allocate(Size, Alignment);
    if (DeviceReadOnly) {
      Context->SharedReadOnlyAllocs.insert(*ResultPtr);
    }
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*ResultPtr),
                                 std::forward_as_tuple(Context));
    }
  } catch (const UsmAllocationException &Ex) {
    *ResultPtr = nullptr;
    return Ex.getError();
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piextUSMHostAlloc(void **ResultPtr, pi_context Context,
                            pi_usm_mem_properties *Properties, size_t Size,
                            pi_uint32 Alignment) {
  // L0 supports alignment up to 64KB and silently ignores higher values.
  // We flag alignment > 64KB as an invalid value.
  if (Alignment > 65536)
    return PI_ERROR_INVALID_VALUE;

  pi_platform Plt = Context->getPlatform();
  // If indirect access tracking is enabled then lock the mutex which is
  // guarding contexts container in the platform. This prevents new kernels from
  // being submitted in any context while we are in the process of allocating a
  // memory, this is needed to properly capture allocations by kernels with
  // indirect access. This lock also protects access to the context's data
  // structures. If indirect access tracking is not enabled then lock context
  // mutex to protect access to context's data structures.
  std::shared_lock<ur_shared_mutex> ContextLock(Context->Mutex,
                                                std::defer_lock);
  std::unique_lock<ur_shared_mutex> IndirectAccessTrackingLock(
      Plt->ContextsMutex, std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    IndirectAccessTrackingLock.lock();
    // We are going to defer memory release if there are kernels with indirect
    // access, that is why explicitly retain context to be sure that it is
    // released after all memory allocations in this context are released.
    PI_CALL(piContextRetain(Context));
  } else {
    ContextLock.lock();
  }

  if (!UseUSMAllocator ||
      // L0 spec says that allocation fails if Alignment != 2^n, in order to
      // keep the same behavior for the allocator, just call L0 API directly and
      // return the error code.
      ((Alignment & (Alignment - 1)) != 0)) {
    pi_result Res =
        USMHostAllocImpl(ResultPtr, Context, Properties, Size, Alignment);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*ResultPtr),
                                 std::forward_as_tuple(Context));
    }
    return Res;
  }

  // There is a single allocator for Host USM allocations, so we don't need to
  // find the allocator depending on context as we do for Shared and Device
  // allocations.
  try {
    *ResultPtr = Context->HostMemAllocContext->allocate(Size, Alignment);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*ResultPtr),
                                 std::forward_as_tuple(Context));
    }
  } catch (const UsmAllocationException &Ex) {
    *ResultPtr = nullptr;
    return Ex.getError();
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

// Helper function to deallocate USM memory, if indirect access support is
// enabled then a caller must lock the platform-level mutex guarding the
// container with contexts because deallocating the memory can turn RefCount of
// a context to 0 and as a result the context being removed from the list of
// tracked contexts.
// If indirect access tracking is not enabled then caller must lock Context
// mutex.
static pi_result USMFreeHelper(pi_context Context, void *Ptr,
                               bool OwnZeMemHandle) {
  if (!OwnZeMemHandle) {
    // Memory should not be freed
    return PI_SUCCESS;
  }

  if (IndirectAccessTrackingEnabled) {
    auto It = Context->MemAllocs.find(Ptr);
    if (It == std::end(Context->MemAllocs)) {
      die("All memory allocations must be tracked!");
    }
    if (!It->second.RefCount.decrementAndTest()) {
      // Memory can't be deallocated yet.
      return PI_SUCCESS;
    }

    // Reference count is zero, it is ok to free memory.
    // We don't need to track this allocation anymore.
    Context->MemAllocs.erase(It);
  }

  if (!UseUSMAllocator) {
    pi_result Res = USMFreeImpl(Context, Ptr);
    if (IndirectAccessTrackingEnabled)
      PI_CALL(ContextReleaseHelper(Context));
    return Res;
  }

  // Query the device of the allocation to determine the right allocator context
  ze_device_handle_t ZeDeviceHandle;
  ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;

  // Query memory type of the pointer we're freeing to determine the correct
  // way to do it(directly or via an allocator)
  auto ZeResult =
      ZE_CALL_NOCHECK(zeMemGetAllocProperties,
                      (Context->ZeContext, Ptr, &ZeMemoryAllocationProperties,
                       &ZeDeviceHandle));

  // Handle the case that L0 RT was already unloaded
  if (ZeResult == ZE_RESULT_ERROR_UNINITIALIZED) {
    if (IndirectAccessTrackingEnabled)
      PI_CALL(ContextReleaseHelper(Context));
    return PI_SUCCESS;
  } else if (ZeResult) {
    return mapError(ZeResult);
  }

  // If memory type is host release from host pool
  if (ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_HOST) {
    try {
      Context->HostMemAllocContext->deallocate(Ptr);
    } catch (const UsmAllocationException &Ex) {
      return Ex.getError();
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
    if (IndirectAccessTrackingEnabled)
      PI_CALL(ContextReleaseHelper(Context));
    return PI_SUCCESS;
  }

  // Points out an allocation in SharedReadOnlyMemAllocContexts
  auto SharedReadOnlyAllocsIterator = Context->SharedReadOnlyAllocs.end();

  if (!ZeDeviceHandle) {
    // The only case where it is OK not have device identified is
    // if the memory is not known to the driver. We should not ever get
    // this either, probably.
    PI_ASSERT(ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_UNKNOWN,
              PI_ERROR_INVALID_DEVICE);
  } else {
    pi_device Device;
    // All context member devices or their descendants are of the same platform.
    auto Platform = Context->getPlatform();
    Device = Platform->getDeviceFromNativeHandle(ZeDeviceHandle);
    PI_ASSERT(Device, PI_ERROR_INVALID_DEVICE);

    auto DeallocationHelper =
        [Context, Device,
         Ptr](std::unordered_map<ze_device_handle_t, USMAllocContext>
                  &AllocContextMap) {
          try {
            auto It = AllocContextMap.find(Device->ZeDevice);
            if (It == AllocContextMap.end())
              return PI_ERROR_INVALID_VALUE;

            // The right context is found, deallocate the pointer
            It->second.deallocate(Ptr);
          } catch (const UsmAllocationException &Ex) {
            return Ex.getError();
          }

          if (IndirectAccessTrackingEnabled)
            PI_CALL(ContextReleaseHelper(Context));
          return PI_SUCCESS;
        };

    switch (ZeMemoryAllocationProperties.type) {
    case ZE_MEMORY_TYPE_SHARED:
      // Distinguish device_read_only allocations since they have own pool.
      SharedReadOnlyAllocsIterator = Context->SharedReadOnlyAllocs.find(Ptr);
      return DeallocationHelper(SharedReadOnlyAllocsIterator !=
                                        Context->SharedReadOnlyAllocs.end()
                                    ? Context->SharedReadOnlyMemAllocContexts
                                    : Context->SharedMemAllocContexts);
    case ZE_MEMORY_TYPE_DEVICE:
      return DeallocationHelper(Context->DeviceMemAllocContexts);
    default:
      // Handled below
      break;
    }
  }

  pi_result Res = USMFreeImpl(Context, Ptr);
  if (SharedReadOnlyAllocsIterator != Context->SharedReadOnlyAllocs.end()) {
    Context->SharedReadOnlyAllocs.erase(SharedReadOnlyAllocsIterator);
  }
  if (IndirectAccessTrackingEnabled)
    PI_CALL(ContextReleaseHelper(Context));
  return Res;
}

pi_result piextUSMFree(pi_context Context, void *Ptr) {
  pi_platform Plt = Context->getPlatform();

  std::scoped_lock<ur_shared_mutex> Lock(
      IndirectAccessTrackingEnabled ? Plt->ContextsMutex : Context->Mutex);

  return USMFreeHelper(Context, Ptr);
}

pi_result piextKernelSetArgPointer(pi_kernel Kernel, pi_uint32 ArgIndex,
                                   size_t ArgSize, const void *ArgValue) {

  PI_CALL(piKernelSetArg(Kernel, ArgIndex, ArgSize, ArgValue));
  return PI_SUCCESS;
}

/// USM Memset API
///
/// @param Queue is the queue to submit to
/// @param Ptr is the ptr to memset
/// @param Value is value to set.  It is interpreted as an 8-bit value and the
/// upper
///        24 bits are ignored
/// @param Count is the size in bytes to memset
/// @param NumEventsInWaitlist is the number of events to wait on
/// @param EventsWaitlist is an array of events to wait on
/// @param Event is the event that represents this operation
pi_result piextUSMEnqueueMemset(pi_queue Queue, void *Ptr, pi_int32 Value,
                                size_t Count, pi_uint32 NumEventsInWaitlist,
                                const pi_event *EventsWaitlist,
                                pi_event *Event) {
  if (!Ptr) {
    return PI_ERROR_INVALID_VALUE;
  }

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  std::scoped_lock<ur_shared_mutex> Lock(Queue->Mutex);
  return enqueueMemFillHelper(
      // TODO: do we need a new command type for USM memset?
      PI_COMMAND_TYPE_MEM_BUFFER_FILL, Queue, Ptr,
      &Value, // It will be interpreted as an 8-bit value,
      1,      // which is indicated with this pattern_size==1
      Count, NumEventsInWaitlist, EventsWaitlist, Event);
}

// Helper function to check if a pointer is a device pointer.
static bool IsDevicePointer(pi_context Context, const void *Ptr) {
  ze_device_handle_t ZeDeviceHandle;
  ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;

  // Query memory type of the pointer
  ZE_CALL(zeMemGetAllocProperties,
          (Context->ZeContext, Ptr, &ZeMemoryAllocationProperties,
           &ZeDeviceHandle));

  return (ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_DEVICE);
}

pi_result piextUSMEnqueueMemcpy(pi_queue Queue, pi_bool Blocking, void *DstPtr,
                                const void *SrcPtr, size_t Size,
                                pi_uint32 NumEventsInWaitlist,
                                const pi_event *EventsWaitlist,
                                pi_event *Event) {

  if (!DstPtr) {
    return PI_ERROR_INVALID_VALUE;
  }

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  // Device to Device copies are found to execute slower on copy engine
  // (versus compute engine).
  bool PreferCopyEngine = !IsDevicePointer(Queue->Context, SrcPtr) ||
                          !IsDevicePointer(Queue->Context, DstPtr);

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyHelper(
      // TODO: do we need a new command type for this?
      PI_COMMAND_TYPE_MEM_BUFFER_COPY, Queue, DstPtr, Blocking, Size, SrcPtr,
      NumEventsInWaitlist, EventsWaitlist, Event, PreferCopyEngine);
}

/// Hint to migrate memory to the device
///
/// @param Queue is the queue to submit to
/// @param Ptr points to the memory to migrate
/// @param Size is the number of bytes to migrate
/// @param Flags is a bitfield used to specify memory migration options
/// @param NumEventsInWaitlist is the number of events to wait on
/// @param EventsWaitlist is an array of events to wait on
/// @param Event is the event that represents this operation
pi_result piextUSMEnqueuePrefetch(pi_queue Queue, const void *Ptr, size_t Size,
                                  pi_usm_migration_flags Flags,
                                  pi_uint32 NumEventsInWaitList,
                                  const pi_event *EventWaitList,
                                  pi_event *OutEvent) {

  // flags is currently unused so fail if set
  PI_ASSERT(Flags == 0, PI_ERROR_INVALID_VALUE);
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  bool UseCopyEngine = false;

  // Please note that the following code should be run before the
  // subsequent getAvailableCommandList() call so that there is no
  // dead-lock from waiting unsubmitted events in an open batch.
  // The createAndRetainPiZeEventList() has the proper side-effect
  // of submitting batches with dependent events.
  //
  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
          NumEventsInWaitList, EventWaitList, Queue, UseCopyEngine))
    return Res;

  // Get a new command list to be used on this call
  pi_command_list_ptr_t CommandList{};
  // TODO: Change UseCopyEngine argument to 'true' once L0 backend
  // support is added
  if (auto Res = Queue->Context->getAvailableCommandList(Queue, CommandList,
                                                         UseCopyEngine))
    return Res;

  // TODO: do we need to create a unique command type for this?
  ze_event_handle_t ZeEvent = nullptr;
  pi_event InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  pi_event *Event = OutEvent ? OutEvent : &InternalEvent;
  auto Res = createEventAndAssociateQueue(Queue, Event, PI_COMMAND_TYPE_USER,
                                          CommandList, IsInternal);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &WaitList = (*Event)->WaitList;
  const auto &ZeCommandList = CommandList->first;
  if (WaitList.Length) {
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }
  // TODO: figure out how to translate "flags"
  ZE_CALL(zeCommandListAppendMemoryPrefetch, (ZeCommandList, Ptr, Size));

  // TODO: Level Zero does not have a completion "event" with the prefetch API,
  // so manually add command to signal our event.
  ZE_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

  if (auto Res = Queue->executeCommandList(CommandList, false))
    return Res;

  return PI_SUCCESS;
}

/// USM memadvise API to govern behavior of automatic migration mechanisms
///
/// @param Queue is the queue to submit to
/// @param Ptr is the data to be advised
/// @param Length is the size in bytes of the meory to advise
/// @param Advice is device specific advice
/// @param Event is the event that represents this operation
///
pi_result piextUSMEnqueueMemAdvise(pi_queue Queue, const void *Ptr,
                                   size_t Length, pi_mem_advice Advice,
                                   pi_event *OutEvent) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  // Lock automatically releases when this goes out of scope.
  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  auto ZeAdvice = ur_cast<ze_memory_advice_t>(Advice);

  bool UseCopyEngine = false;

  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(0, nullptr, Queue,
                                                          UseCopyEngine))
    return Res;

  // Get a new command list to be used on this call
  pi_command_list_ptr_t CommandList{};
  // UseCopyEngine is set to 'false' here.
  // TODO: Additional analysis is required to check if this operation will
  // run faster on copy engines.
  if (auto Res = Queue->Context->getAvailableCommandList(Queue, CommandList,
                                                         UseCopyEngine))
    return Res;

  // TODO: do we need to create a unique command type for this?
  ze_event_handle_t ZeEvent = nullptr;
  pi_event InternalEvent;
  bool IsInternal = OutEvent == nullptr;
  pi_event *Event = OutEvent ? OutEvent : &InternalEvent;
  auto Res = createEventAndAssociateQueue(Queue, Event, PI_COMMAND_TYPE_USER,
                                          CommandList, IsInternal);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  if (WaitList.Length) {
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }

  ZE_CALL(zeCommandListAppendMemAdvise,
          (ZeCommandList, Queue->Device->ZeDevice, Ptr, Length, ZeAdvice));

  // TODO: Level Zero does not have a completion "event" with the advise API,
  // so manually add command to signal our event.
  ZE_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

  Queue->executeCommandList(CommandList, false);

  return PI_SUCCESS;
}

/// USM 2D Fill API
///
/// \param queue is the queue to submit to
/// \param ptr is the ptr to fill
/// \param pitch is the total width of the destination memory including padding
/// \param pattern is a pointer with the bytes of the pattern to set
/// \param pattern_size is the size in bytes of the pattern
/// \param width is width in bytes of each row to fill
/// \param height is height the columns to fill
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
__SYCL_EXPORT pi_result piextUSMEnqueueFill2D(pi_queue queue, void *ptr,
                                              size_t pitch, size_t pattern_size,
                                              const void *pattern, size_t width,
                                              size_t height,
                                              pi_uint32 num_events_in_waitlist,
                                              const pi_event *events_waitlist,
                                              pi_event *event) {
  std::ignore = queue;
  std::ignore = ptr;
  std::ignore = pitch;
  std::ignore = pattern_size;
  std::ignore = pattern;
  std::ignore = width;
  std::ignore = height;
  std::ignore = num_events_in_waitlist;
  std::ignore = events_waitlist;
  std::ignore = event;
  die("piextUSMEnqueueFill2D: not implemented");
  return {};
}

/// USM 2D Memset API
///
/// \param queue is the queue to submit to
/// \param ptr is the ptr to fill
/// \param pitch is the total width of the destination memory including padding
/// \param pattern is a pointer with the bytes of the pattern to set
/// \param pattern_size is the size in bytes of the pattern
/// \param width is width in bytes of each row to fill
/// \param height is height the columns to fill
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
__SYCL_EXPORT pi_result piextUSMEnqueueMemset2D(
    pi_queue queue, void *ptr, size_t pitch, int value, size_t width,
    size_t height, pi_uint32 num_events_in_waitlist,
    const pi_event *events_waitlist, pi_event *event) {
  std::ignore = queue;
  std::ignore = ptr;
  std::ignore = pitch;
  std::ignore = value;
  std::ignore = width;
  std::ignore = height;
  std::ignore = num_events_in_waitlist;
  std::ignore = events_waitlist;
  std::ignore = event;
  die("piextUSMEnqueueMemset2D: not implemented");
  return {};
}

/// USM 2D Memcpy API
///
/// \param queue is the queue to submit to
/// \param blocking is whether this operation should block the host
/// \param dst_ptr is the location the data will be copied
/// \param dst_pitch is the total width of the destination memory including
/// padding
/// \param src_ptr is the data to be copied
/// \param dst_pitch is the total width of the source memory including padding
/// \param width is width in bytes of each row to be copied
/// \param height is height the columns to be copied
/// \param num_events_in_waitlist is the number of events to wait on
/// \param events_waitlist is an array of events to wait on
/// \param event is the event that represents this operation
__SYCL_EXPORT pi_result piextUSMEnqueueMemcpy2D(
    pi_queue Queue, pi_bool Blocking, void *DstPtr, size_t DstPitch,
    const void *SrcPtr, size_t SrcPitch, size_t Width, size_t Height,
    pi_uint32 NumEventsInWaitlist, const pi_event *EventWaitlist,
    pi_event *Event) {
  if (!DstPtr || !SrcPtr)
    return PI_ERROR_INVALID_VALUE;

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  pi_buff_rect_offset_struct ZeroOffset{0, 0, 0};
  pi_buff_rect_region_struct Region{Width, Height, 0};

  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  // Device to Device copies are found to execute slower on copy engine
  // (versus compute engine).
  bool PreferCopyEngine = !IsDevicePointer(Queue->Context, SrcPtr) ||
                          !IsDevicePointer(Queue->Context, DstPtr);

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyRectHelper(
      // TODO: do we need a new command type for this?
      PI_COMMAND_TYPE_MEM_BUFFER_COPY_RECT, Queue, SrcPtr, DstPtr, &ZeroOffset,
      &ZeroOffset, &Region, SrcPitch, DstPitch, /*SrcSlicePitch=*/0,
      /*DstSlicePitch=*/0, Blocking, NumEventsInWaitlist, EventWaitlist, Event,
      PreferCopyEngine);
}

/// API to query information about USM allocated pointers.
/// Valid Queries:
///   PI_MEM_ALLOC_TYPE returns host/device/shared pi_usm_type value
///   PI_MEM_ALLOC_BASE_PTR returns the base ptr of an allocation if
///                         the queried pointer fell inside an allocation.
///                         Result must fit in void *
///   PI_MEM_ALLOC_SIZE returns how big the queried pointer's
///                     allocation is in bytes. Result is a size_t.
///   PI_MEM_ALLOC_DEVICE returns the pi_device this was allocated against
///
/// @param Context is the pi_context
/// @param Ptr is the pointer to query
/// @param ParamName is the type of query to perform
/// @param ParamValueSize is the size of the result in bytes
/// @param ParamValue is the result
/// @param ParamValueRet is how many bytes were written
pi_result piextUSMGetMemAllocInfo(pi_context Context, const void *Ptr,
                                  pi_mem_alloc_info ParamName,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {
  PI_ASSERT(Context, PI_ERROR_INVALID_CONTEXT);

  ze_device_handle_t ZeDeviceHandle;
  ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;

  ZE_CALL(zeMemGetAllocProperties,
          (Context->ZeContext, Ptr, &ZeMemoryAllocationProperties,
           &ZeDeviceHandle));

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_MEM_ALLOC_TYPE: {
    pi_usm_type MemAllocaType;
    switch (ZeMemoryAllocationProperties.type) {
    case ZE_MEMORY_TYPE_UNKNOWN:
      MemAllocaType = PI_MEM_TYPE_UNKNOWN;
      break;
    case ZE_MEMORY_TYPE_HOST:
      MemAllocaType = PI_MEM_TYPE_HOST;
      break;
    case ZE_MEMORY_TYPE_DEVICE:
      MemAllocaType = PI_MEM_TYPE_DEVICE;
      break;
    case ZE_MEMORY_TYPE_SHARED:
      MemAllocaType = PI_MEM_TYPE_SHARED;
      break;
    default:
      urPrint("piextUSMGetMemAllocInfo: unexpected usm memory type\n");
      return PI_ERROR_INVALID_VALUE;
    }
    return ReturnValue(MemAllocaType);
  }
  case PI_MEM_ALLOC_DEVICE:
    if (ZeDeviceHandle) {
      auto Platform = Context->getPlatform();
      auto Device = Platform->getDeviceFromNativeHandle(ZeDeviceHandle);
      return Device ? ReturnValue(Device) : PI_ERROR_INVALID_VALUE;
    } else {
      return PI_ERROR_INVALID_VALUE;
    }
  case PI_MEM_ALLOC_BASE_PTR: {
    void *Base;
    ZE_CALL(zeMemGetAddressRange, (Context->ZeContext, Ptr, &Base, nullptr));
    return ReturnValue(Base);
  }
  case PI_MEM_ALLOC_SIZE: {
    size_t Size;
    ZE_CALL(zeMemGetAddressRange, (Context->ZeContext, Ptr, nullptr, &Size));
    return ReturnValue(Size);
  }
  default:
    urPrint("piextUSMGetMemAllocInfo: unsupported ParamName\n");
    return PI_ERROR_INVALID_VALUE;
  }
  return PI_SUCCESS;
}

/// API for writing data from host to a device global variable.
///
/// \param Queue is the queue
/// \param Program is the program containing the device global variable
/// \param Name is the unique identifier for the device global variable
/// \param BlockingWrite is true if the write should block
/// \param Count is the number of bytes to copy
/// \param Offset is the byte offset into the device global variable to start
/// copying
/// \param Src is a pointer to where the data must be copied from
/// \param NumEventsInWaitList is a number of events in the wait list
/// \param EventWaitList is the wait list
/// \param Event is the resulting event
pi_result piextEnqueueDeviceGlobalVariableWrite(
    pi_queue Queue, pi_program Program, const char *Name, pi_bool BlockingWrite,
    size_t Count, size_t Offset, const void *Src, pi_uint32 NumEventsInWaitList,
    const pi_event *EventsWaitList, pi_event *Event) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  // Find global variable pointer
  size_t GlobalVarSize = 0;
  void *GlobalVarPtr = nullptr;
  ZE_CALL(zeModuleGetGlobalPointer,
          (Program->ZeModule, Name, &GlobalVarSize, &GlobalVarPtr));
  if (GlobalVarSize < Offset + Count) {
    setErrorMessage("Write device global variable is out of range.",
                    UR_RESULT_ERROR_INVALID_VALUE);
    return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
  }

  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = !IsDevicePointer(Queue->Context, Src);

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyHelper(PI_COMMAND_TYPE_DEVICE_GLOBAL_VARIABLE_WRITE,
                              Queue, ur_cast<char *>(GlobalVarPtr) + Offset,
                              BlockingWrite, Count, Src, NumEventsInWaitList,
                              EventsWaitList, Event, PreferCopyEngine);
}

/// API reading data from a device global variable to host.
///
/// \param Queue is the queue
/// \param Program is the program containing the device global variable
/// \param Name is the unique identifier for the device global variable
/// \param BlockingRead is true if the read should block
/// \param Count is the number of bytes to copy
/// \param Offset is the byte offset into the device global variable to start
/// copying
/// \param Dst is a pointer to where the data must be copied to
/// \param NumEventsInWaitList is a number of events in the wait list
/// \param EventWaitList is the wait list
/// \param Event is the resulting event
pi_result piextEnqueueDeviceGlobalVariableRead(
    pi_queue Queue, pi_program Program, const char *Name, pi_bool BlockingRead,
    size_t Count, size_t Offset, void *Dst, pi_uint32 NumEventsInWaitList,
    const pi_event *EventsWaitList, pi_event *Event) {
  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  std::scoped_lock<ur_shared_mutex> lock(Queue->Mutex);

  // Find global variable pointer
  size_t GlobalVarSize = 0;
  void *GlobalVarPtr = nullptr;
  ZE_CALL(zeModuleGetGlobalPointer,
          (Program->ZeModule, Name, &GlobalVarSize, &GlobalVarPtr));
  if (GlobalVarSize < Offset + Count) {
    setErrorMessage("Read from device global variable is out of range.",
                    UR_RESULT_ERROR_INVALID_VALUE);
    return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
  }

  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = !IsDevicePointer(Queue->Context, Dst);

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyHelper(
      PI_COMMAND_TYPE_DEVICE_GLOBAL_VARIABLE_READ, Queue, Dst, BlockingRead,
      Count, ur_cast<char *>(GlobalVarPtr) + Offset, NumEventsInWaitList,
      EventsWaitList, Event, PreferCopyEngine);
}
/// API for Read from host pipe.
///
/// \param Queue is the queue
/// \param Program is the program containing the device variable
/// \param PipeSymbol is the unique identifier for the device variable
/// \param Blocking is true if the write should block
/// \param Ptr is a pointer to where the data will be copied to
/// \param Size is size of the data that is read/written from/to pipe
/// \param NumEventsInWaitList is a number of events in the wait list
/// \param EventWaitList is the wait list
/// \param Event is the resulting event
pi_result piextEnqueueReadHostPipe(pi_queue Queue, pi_program Program,
                                   const char *PipeSymbol, pi_bool Blocking,
                                   void *Ptr, size_t Size,
                                   pi_uint32 NumEventsInWaitList,
                                   const pi_event *EventsWaitList,
                                   pi_event *Event) {
  (void)Queue;
  (void)Program;
  (void)PipeSymbol;
  (void)Blocking;
  (void)Ptr;
  (void)Size;
  (void)NumEventsInWaitList;
  (void)EventsWaitList;
  (void)Event;

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  die("piextEnqueueReadHostPipe: not implemented");
  return {};
}

/// API for write to pipe of a given name.
///
/// \param Queue is the queue
/// \param Program is the program containing the device variable
/// \param PipeSymbol is the unique identifier for the device variable
/// \param Blocking is true if the write should block
/// \param Ptr is a pointer to where the data must be copied from
/// \param Size is size of the data that is read/written from/to pipe
/// \param NumEventsInWaitList is a number of events in the wait list
/// \param EventWaitList is the wait list
/// \param Event is the resulting event
pi_result piextEnqueueWriteHostPipe(pi_queue Queue, pi_program Program,
                                    const char *PipeSymbol, pi_bool Blocking,
                                    void *Ptr, size_t Size,
                                    pi_uint32 NumEventsInWaitList,
                                    const pi_event *EventsWaitList,
                                    pi_event *Event) {
  (void)Queue;
  (void)Program;
  (void)PipeSymbol;
  (void)Blocking;
  (void)Ptr;
  (void)Size;
  (void)NumEventsInWaitList;
  (void)EventsWaitList;
  (void)Event;

  PI_ASSERT(Queue, PI_ERROR_INVALID_QUEUE);

  die("piextEnqueueWriteHostPipe: not implemented");
  return {};
}

pi_result piKernelSetExecInfo(pi_kernel Kernel, pi_kernel_exec_info ParamName,
                              size_t ParamValueSize, const void *ParamValue) {
  (void)ParamValueSize;
  PI_ASSERT(Kernel, PI_ERROR_INVALID_KERNEL);
  PI_ASSERT(ParamValue, PI_ERROR_INVALID_VALUE);

  std::scoped_lock<ur_shared_mutex> Guard(Kernel->Mutex);
  if (ParamName == PI_USM_INDIRECT_ACCESS &&
      *(static_cast<const pi_bool *>(ParamValue)) == PI_TRUE) {
    // The whole point for users really was to not need to know anything
    // about the types of allocations kernel uses. So in DPC++ we always
    // just set all 3 modes for each kernel.
    ze_kernel_indirect_access_flags_t IndirectFlags =
        ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST |
        ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE |
        ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
    ZE_CALL(zeKernelSetIndirectAccess, (Kernel->ZeKernel, IndirectFlags));
  } else if (ParamName == PI_EXT_KERNEL_EXEC_INFO_CACHE_CONFIG) {
    ze_cache_config_flag_t ZeCacheConfig;
    switch (*(static_cast<const pi_kernel_cache_config *>(ParamValue))) {
    case PI_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_SLM:
      ZeCacheConfig = ZE_CACHE_CONFIG_FLAG_LARGE_SLM;
      break;
    case PI_EXT_KERNEL_EXEC_INFO_CACHE_LARGE_DATA:
      ZeCacheConfig = ZE_CACHE_CONFIG_FLAG_LARGE_DATA;
      break;
    case PI_EXT_KERNEL_EXEC_INFO_CACHE_DEFAULT:
      ZeCacheConfig = static_cast<ze_cache_config_flag_t>(0);
      break;
    default:
      // Unexpected cache configuration value.
      return PI_ERROR_INVALID_VALUE;
    }
    ZE_CALL(zeKernelSetCacheConfig, (Kernel->ZeKernel, ZeCacheConfig););
  } else {
    urPrint("piKernelSetExecInfo: unsupported ParamName\n");
    return PI_ERROR_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

pi_result piextProgramSetSpecializationConstant(pi_program Prog,
                                                pi_uint32 SpecID, size_t,
                                                const void *SpecValue) {
  std::scoped_lock<ur_shared_mutex> Guard(Prog->Mutex);

  // Remember the value of this specialization constant until the program is
  // built.  Note that we only save the pointer to the buffer that contains the
  // value.  The caller is responsible for maintaining storage for this buffer.
  //
  // NOTE: SpecSize is unused in Level Zero, the size is known from SPIR-V by
  // SpecID.
  Prog->SpecConstants[SpecID] = SpecValue;

  return PI_SUCCESS;
}

const char SupportedVersion[] = _PI_LEVEL_ZERO_PLUGIN_VERSION_STRING;

pi_result piPluginInit(pi_plugin *PluginInit) {
  PI_ASSERT(PluginInit, PI_ERROR_INVALID_VALUE);

  // Check that the major version matches in PiVersion and SupportedVersion
  _PI_PLUGIN_VERSION_CHECK(PluginInit->PiVersion, SupportedVersion);

  // TODO: handle versioning/targets properly.
  size_t PluginVersionSize = sizeof(PluginInit->PluginVersion);

  PI_ASSERT(strlen(_PI_LEVEL_ZERO_PLUGIN_VERSION_STRING) < PluginVersionSize,
            PI_ERROR_INVALID_VALUE);

  strncpy(PluginInit->PluginVersion, SupportedVersion, PluginVersionSize);

#define _PI_API(api)                                                           \
  (PluginInit->PiFunctionTable).api = (decltype(&::api))(&api);
#include <sycl/detail/pi.def>

  enableZeTracing();
  return PI_SUCCESS;
}

pi_result piextPluginGetOpaqueData(void *opaque_data_param,
                                   void **opaque_data_return) {
  (void)opaque_data_param;
  (void)opaque_data_return;
  return PI_ERROR_UNKNOWN;
}

// SYCL RT calls this api to notify the end of plugin lifetime.
// Windows: dynamically loaded plugins might have been unloaded already
// when this is called. Sycl RT holds onto the PI plugin so it can be
// called safely. But this is not transitive. If the PI plugin in turn
// dynamically loaded a different DLL, that may have been unloaded.
// It can include all the jobs to tear down resources before
// the plugin is unloaded from memory.
pi_result piTearDown(void *PluginParameter) {
  (void)PluginParameter;
  bool LeakFound = false;
  // reclaim pi_platform objects here since we don't have piPlatformRelease.
  for (pi_platform Platform : *PiPlatformsCache) {
    delete Platform;
  }
  delete PiPlatformsCache;
  delete PiPlatformsCacheMutex;

  // Print the balance of various create/destroy native calls.
  // The idea is to verify if the number of create(+) and destroy(-) calls are
  // matched.
  if (UrL0Debug & UR_L0_DEBUG_CALL_COUNT) {
    // clang-format off
    //
    // The format of this table is such that each row accounts for a
    // specific type of objects, and all elements in the raw except the last
    // one are allocating objects of that type, while the last element is known
    // to deallocate objects of that type.
    //
    std::vector<std::vector<const char *>> CreateDestroySet = {
      {"zeContextCreate",      "zeContextDestroy"},
      {"zeCommandQueueCreate", "zeCommandQueueDestroy"},
      {"zeModuleCreate",       "zeModuleDestroy"},
      {"zeKernelCreate",       "zeKernelDestroy"},
      {"zeEventPoolCreate",    "zeEventPoolDestroy"},
      {"zeCommandListCreateImmediate", "zeCommandListCreate", "zeCommandListDestroy"},
      {"zeEventCreate",        "zeEventDestroy"},
      {"zeFenceCreate",        "zeFenceDestroy"},
      {"zeImageCreate",        "zeImageDestroy"},
      {"zeSamplerCreate",      "zeSamplerDestroy"},
      {"zeMemAllocDevice", "zeMemAllocHost", "zeMemAllocShared", "zeMemFree"},
    };

    // A sample output aimed below is this:
    // ------------------------------------------------------------------------
    //                zeContextCreate = 1     \--->        zeContextDestroy = 1
    //           zeCommandQueueCreate = 1     \--->   zeCommandQueueDestroy = 1
    //                 zeModuleCreate = 1     \--->         zeModuleDestroy = 1
    //                 zeKernelCreate = 1     \--->         zeKernelDestroy = 1
    //              zeEventPoolCreate = 1     \--->      zeEventPoolDestroy = 1
    //   zeCommandListCreateImmediate = 1     |
    //            zeCommandListCreate = 1     \--->    zeCommandListDestroy = 1  ---> LEAK = 1
    //                  zeEventCreate = 2     \--->          zeEventDestroy = 2
    //                  zeFenceCreate = 1     \--->          zeFenceDestroy = 1
    //                  zeImageCreate = 0     \--->          zeImageDestroy = 0
    //                zeSamplerCreate = 0     \--->        zeSamplerDestroy = 0
    //               zeMemAllocDevice = 0     |
    //                 zeMemAllocHost = 1     |
    //               zeMemAllocShared = 0     \--->               zeMemFree = 1
    //
    // clang-format on

    fprintf(stderr, "ZE_DEBUG=%d: check balance of create/destroy calls\n",
            UR_L0_DEBUG_CALL_COUNT);
    fprintf(stderr,
            "----------------------------------------------------------\n");
    for (const auto &Row : CreateDestroySet) {
      int diff = 0;
      for (auto I = Row.begin(); I != Row.end();) {
        const char *ZeName = *I;
        const auto &ZeCount = (*ZeCallCount)[*I];

        bool First = (I == Row.begin());
        bool Last = (++I == Row.end());

        if (Last) {
          fprintf(stderr, " \\--->");
          diff -= ZeCount;
        } else {
          diff += ZeCount;
          if (!First) {
            fprintf(stderr, " | \n");
          }
        }

        fprintf(stderr, "%30s = %-5d", ZeName, ZeCount);
      }

      if (diff) {
        LeakFound = true;
        fprintf(stderr, " ---> LEAK = %d", diff);
      }
      fprintf(stderr, "\n");
    }

    ZeCallCount->clear();
    delete ZeCallCount;
    ZeCallCount = nullptr;
  }
  if (LeakFound)
    return PI_ERROR_INVALID_MEM_OBJECT;

  disableZeTracing();
  return PI_SUCCESS;
}

pi_result _pi_buffer::getZeHandlePtr(char **&ZeHandlePtr,
                                     access_mode_t AccessMode,
                                     pi_device Device) {
  char *ZeHandle;
  PI_CALL(getZeHandle(ZeHandle, AccessMode, Device));
  ZeHandlePtr = &Allocations[Device].ZeHandle;
  return PI_SUCCESS;
}

size_t _pi_buffer::getAlignment() const {
  // Choose an alignment that is at most 64 and is the next power of 2
  // for sizes less than 64.
  auto Alignment = Size;
  if (Alignment > 32UL)
    Alignment = 64UL;
  else if (Alignment > 16UL)
    Alignment = 32UL;
  else if (Alignment > 8UL)
    Alignment = 16UL;
  else if (Alignment > 4UL)
    Alignment = 8UL;
  else if (Alignment > 2UL)
    Alignment = 4UL;
  else if (Alignment > 1UL)
    Alignment = 2UL;
  else
    Alignment = 1UL;
  return Alignment;
}

pi_result _pi_buffer::getZeHandle(char *&ZeHandle, access_mode_t AccessMode,
                                  pi_device Device) {

  // NOTE: There might be no valid allocation at all yet and we get
  // here from piEnqueueKernelLaunch that would be doing the buffer
  // initialization. In this case the Device is not null as kernel
  // launch is always on a specific device.
  if (!Device)
    Device = LastDeviceWithValidAllocation;
  // If the device is still not selected then use the first one in
  // the context of the buffer.
  if (!Device)
    Device = Context->Devices[0];

  auto &Allocation = Allocations[Device];

  // Sub-buffers don't maintain own allocations but rely on parent buffer.
  if (isSubBuffer()) {
    PI_CALL(SubBuffer.Parent->getZeHandle(ZeHandle, AccessMode, Device));
    ZeHandle += SubBuffer.Origin;
    // Still store the allocation info in the PI sub-buffer for
    // getZeHandlePtr to work. At least zeKernelSetArgumentValue needs to
    // be given a pointer to the allocation handle rather than its value.
    //
    Allocation.ZeHandle = ZeHandle;
    Allocation.ReleaseAction = allocation_t::keep;
    LastDeviceWithValidAllocation = Device;
    return PI_SUCCESS;
  }

  // First handle case where the buffer is represented by only
  // a single host allocation.
  if (OnHost) {
    auto &HostAllocation = Allocations[nullptr];
    // The host allocation may already exists, e.g. with imported
    // host ptr, or in case of interop buffer.
    if (!HostAllocation.ZeHandle) {
      if (USMAllocatorConfigInstance.EnableBuffers) {
        HostAllocation.ReleaseAction = allocation_t::free;
        PI_CALL(piextUSMHostAlloc(ur_cast<void **>(&ZeHandle), Context, nullptr,
                                  Size, getAlignment()));
      } else {
        HostAllocation.ReleaseAction = allocation_t::free_native;
        PI_CALL(
            ZeHostMemAllocHelper(ur_cast<void **>(&ZeHandle), Context, Size));
      }
      HostAllocation.ZeHandle = ZeHandle;
      HostAllocation.Valid = true;
    }
    Allocation = HostAllocation;
    Allocation.ReleaseAction = allocation_t::keep;
    ZeHandle = Allocation.ZeHandle;
    LastDeviceWithValidAllocation = Device;
    return PI_SUCCESS;
  }
  // Reads user setting on how to deal with buffers in contexts where
  // all devices have the same root-device. Returns "true" if the
  // preference is to have allocate on each [sub-]device and migrate
  // normally (copy) to other sub-devices as needed. Returns "false"
  // if the preference is to have single root-device allocations
  // serve the needs of all [sub-]devices, meaning potentially more
  // cross-tile traffic.
  //
  static const bool SingleRootDeviceBufferMigration = [] {
    const char *UrRet =
        std::getenv("UR_L0_SINGLE_ROOT_DEVICE_BUFFER_MIGRATION");
    const char *PiRet =
        std::getenv("SYCL_PI_LEVEL_ZERO_SINGLE_ROOT_DEVICE_BUFFER_MIGRATION");
    const char *EnvStr = UrRet ? UrRet : (PiRet ? PiRet : nullptr);

    if (EnvStr)
      return (std::stoi(EnvStr) != 0);
    // The default is to migrate normally, which may not always be the
    // best option (depends on buffer access patterns), but is an
    // overall win on the set of the available benchmarks.
    return true;
  }();

  // Peform actual device allocation as needed.
  if (!Allocation.ZeHandle) {
    if (!SingleRootDeviceBufferMigration && Context->SingleRootDevice &&
        Context->SingleRootDevice != Device) {
      // If all devices in the context are sub-devices of the same device
      // then we reuse root-device allocation by all sub-devices in the
      // context.
      // TODO: we can probably generalize this and share root-device
      //       allocations by its own sub-devices even if not all other
      //       devices in the context have the same root.
      PI_CALL(getZeHandle(ZeHandle, AccessMode, Context->SingleRootDevice));
      Allocation.ReleaseAction = allocation_t::keep;
      Allocation.ZeHandle = ZeHandle;
      Allocation.Valid = true;
      return PI_SUCCESS;
    } else { // Create device allocation
      if (USMAllocatorConfigInstance.EnableBuffers) {
        Allocation.ReleaseAction = allocation_t::free;
        PI_CALL(piextUSMDeviceAlloc(ur_cast<void **>(&ZeHandle), Context,
                                    Device, nullptr, Size, getAlignment()));
      } else {
        Allocation.ReleaseAction = allocation_t::free_native;
        PI_CALL(ZeDeviceMemAllocHelper(ur_cast<void **>(&ZeHandle), Context,
                                       Device, Size));
      }
    }
    Allocation.ZeHandle = ZeHandle;
  } else {
    ZeHandle = Allocation.ZeHandle;
  }

  // If some prior access invalidated this allocation then make it valid again.
  if (!Allocation.Valid) {
    // LastDeviceWithValidAllocation should always have valid allocation.
    if (Device == LastDeviceWithValidAllocation)
      die("getZeHandle: last used allocation is not valid");

    // For write-only access the allocation contents is not going to be used.
    // So don't do anything to make it "valid".
    bool NeedCopy = AccessMode != _pi_mem::write_only;
    // It's also possible that the buffer doesn't have a valid allocation
    // yet presumably when it is passed to a kernel that will perform
    // it's intialization.
    if (NeedCopy && !LastDeviceWithValidAllocation) {
      NeedCopy = false;
    }
    char *ZeHandleSrc = nullptr;
    if (NeedCopy) {
      PI_CALL(getZeHandle(ZeHandleSrc, _pi_mem::read_only,
                          LastDeviceWithValidAllocation));
      // It's possible with the single root-device contexts that
      // the buffer is represented by the single root-device
      // allocation and then skip the copy to itself.
      if (ZeHandleSrc == ZeHandle)
        NeedCopy = false;
    }

    if (NeedCopy) {
      // Copy valid buffer data to this allocation.
      // TODO: see if we should better use peer's device allocation used
      // directly, if that capability is reported with zeDeviceCanAccessPeer,
      // instead of maintaining a separate allocation and performing
      // explciit copies.
      //
      // zeCommandListAppendMemoryCopy must not be called from simultaneous
      // threads with the same command list handle, so we need exclusive lock.
      ze_bool_t P2P = false;
      ZE_CALL(
          zeDeviceCanAccessPeer,
          (Device->ZeDevice, LastDeviceWithValidAllocation->ZeDevice, &P2P));
      if (!P2P) {
        // P2P copy is not possible, so copy through the host.
        auto &HostAllocation = Allocations[nullptr];
        // The host allocation may already exists, e.g. with imported
        // host ptr, or in case of interop buffer.
        if (!HostAllocation.ZeHandle) {
          void *ZeHandleHost;
          if (USMAllocatorConfigInstance.EnableBuffers) {
            HostAllocation.ReleaseAction = allocation_t::free;
            PI_CALL(piextUSMHostAlloc(&ZeHandleHost, Context, nullptr, Size,
                                      getAlignment()));
          } else {
            HostAllocation.ReleaseAction = allocation_t::free_native;
            PI_CALL(ZeHostMemAllocHelper(&ZeHandleHost, Context, Size));
          }
          HostAllocation.ZeHandle = ur_cast<char *>(ZeHandleHost);
          HostAllocation.Valid = false;
        }
        std::scoped_lock<ur_mutex> Lock(Context->ImmediateCommandListMutex);
        if (!HostAllocation.Valid) {
          ZE_CALL(zeCommandListAppendMemoryCopy,
                  (Context->ZeCommandListInit,
                   HostAllocation.ZeHandle /* Dst */, ZeHandleSrc, Size,
                   nullptr, 0, nullptr));
          // Mark the host allocation data  as valid so it can be reused.
          // It will be invalidated below if the current access is not
          // read-only.
          HostAllocation.Valid = true;
        }
        ZE_CALL(zeCommandListAppendMemoryCopy,
                (Context->ZeCommandListInit, ZeHandle /* Dst */,
                 HostAllocation.ZeHandle, Size, nullptr, 0, nullptr));
      } else {
        // Perform P2P copy.
        std::scoped_lock<ur_mutex> Lock(Context->ImmediateCommandListMutex);
        ZE_CALL(zeCommandListAppendMemoryCopy,
                (Context->ZeCommandListInit, ZeHandle /* Dst */, ZeHandleSrc,
                 Size, nullptr, 0, nullptr));
      }
    }
    Allocation.Valid = true;
    LastDeviceWithValidAllocation = Device;
  }

  // Invalidate other allocations that would become not valid if
  // this access is not read-only.
  if (AccessMode != _pi_mem::read_only) {
    for (auto &Alloc : Allocations) {
      if (Alloc.first != LastDeviceWithValidAllocation)
        Alloc.second.Valid = false;
    }
  }

  urPrint("getZeHandle(pi_device{%p}) = %p\n", (void *)Device,
          (void *)Allocation.ZeHandle);
  return PI_SUCCESS;
}

pi_result _pi_buffer::free() {
  for (auto &Alloc : Allocations) {
    auto &ZeHandle = Alloc.second.ZeHandle;
    // It is possible that the real allocation wasn't made if the buffer
    // wasn't really used in this location.
    if (!ZeHandle)
      continue;

    switch (Alloc.second.ReleaseAction) {
    case allocation_t::keep:
      break;
    case allocation_t::free: {
      pi_platform Plt = Context->getPlatform();
      std::scoped_lock<ur_shared_mutex> Lock(
          IndirectAccessTrackingEnabled ? Plt->ContextsMutex : Context->Mutex);

      PI_CALL(USMFreeHelper(Context, ZeHandle));
      break;
    }
    case allocation_t::free_native:
      PI_CALL(ZeMemFreeHelper(Context, ZeHandle));
      break;
    case allocation_t::unimport:
      ZeUSMImport.doZeUSMRelease(Context->getPlatform()->ZeDriver, ZeHandle);
      break;
    default:
      die("_pi_buffer::free(): Unhandled release action");
    }
    ZeHandle = nullptr; // don't leave hanging pointers
  }
  return PI_SUCCESS;
}

pi_result piGetDeviceAndHostTimer(pi_device Device, uint64_t *DeviceTime,
                                  uint64_t *HostTime) {
  const uint64_t &ZeTimerResolution =
      Device->ZeDeviceProperties->timerResolution;
  const uint64_t TimestampMaxCount =
      ((1ULL << Device->ZeDeviceProperties->kernelTimestampValidBits) - 1ULL);
  uint64_t DeviceClockCount, Dummy;

  ZE_CALL(zeDeviceGetGlobalTimestamps,
          (Device->ZeDevice, HostTime == nullptr ? &Dummy : HostTime,
           &DeviceClockCount));

  if (DeviceTime != nullptr) {
    *DeviceTime = (DeviceClockCount & TimestampMaxCount) * ZeTimerResolution;
  }
  return PI_SUCCESS;
}

#ifdef _WIN32
#define __SYCL_PLUGIN_DLL_NAME "pi_level_zero.dll"
#include "../common_win_pi_trace/common_win_pi_trace.hpp"
#undef __SYCL_PLUGIN_DLL_NAME
#endif
} // extern "C"
