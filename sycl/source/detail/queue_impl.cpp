//==------------------ queue_impl.cpp - SYCL queue -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/event_deps.hpp>
#include <detail/event_impl.hpp>
#include <detail/memory_manager.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/device.hpp>

#include <cstring>
#include <utility>

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include <detail/xpti_registry.hpp>
#include <sstream>
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {
// Treat 0 as reserved for host task traces
std::atomic<unsigned long long> queue_impl::MNextAvailableQueueID = 1;

thread_local bool NestedCallsDetector = false;
class NestedCallsTracker {
public:
  NestedCallsTracker() {
    if (NestedCallsDetectorRef)
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Calls to sycl::queue::submit cannot be nested. Command group "
          "function objects should use the sycl::handler API instead.");
    NestedCallsDetectorRef = true;
  }

  ~NestedCallsTracker() { NestedCallsDetectorRef = false; }

private:
  // Cache the TLS location to decrease amount of TLS accesses.
  bool &NestedCallsDetectorRef = NestedCallsDetector;
};

static std::vector<ur_event_handle_t>
getUrEvents(const std::vector<sycl::event> &DepEvents) {
  std::vector<ur_event_handle_t> RetUrEvents;
  for (const sycl::event &Event : DepEvents) {
    event_impl &EventImpl = *detail::getSyclObjImpl(Event);
    auto Handle = EventImpl.getHandle();
    if (Handle != nullptr)
      RetUrEvents.push_back(Handle);
  }
  return RetUrEvents;
}

template <>
uint32_t queue_impl::get_info<info::queue::reference_count>() const {
  ur_result_t result = UR_RESULT_SUCCESS;
  getAdapter().call<UrApiKind::urQueueGetInfo>(
      MQueue, UR_QUEUE_INFO_REFERENCE_COUNT, sizeof(result), &result, nullptr);
  return result;
}

template <> context queue_impl::get_info<info::queue::context>() const {
  return get_context();
}

template <> device queue_impl::get_info<info::queue::device>() const {
  return get_device();
}

static event
prepareSYCLEventAssociatedWithQueue(detail::queue_impl &QueueImpl) {
  auto EventImpl = detail::event_impl::create_device_event(QueueImpl);
  EventImpl->setContextImpl(QueueImpl.getContextImpl());
  EventImpl->setStateIncomplete();
  return detail::createSyclObjFromImpl<event>(EventImpl);
}

const std::vector<event> &
queue_impl::getExtendDependencyList(const std::vector<event> &DepEvents,
                                    std::vector<event> &MutableVec,
                                    std::unique_lock<std::mutex> &QueueLock) {
  if (!isInOrder())
    return DepEvents;

  QueueLock.lock();
  EventImplPtr ExtraEvent = MGraph.expired() ? MDefaultGraphDeps.LastEventPtr
                                             : MExtGraphDeps.LastEventPtr;
  std::optional<event> ExternalEvent = popExternalEvent();

  if (!ExternalEvent && !ExtraEvent)
    return DepEvents;

  MutableVec = DepEvents;
  if (ExternalEvent)
    MutableVec.push_back(*ExternalEvent);
  if (ExtraEvent)
    MutableVec.push_back(detail::createSyclObjFromImpl<event>(ExtraEvent));
  return MutableVec;
}

event queue_impl::memset(void *Ptr, int Value, size_t Count,
                         const std::vector<event> &DepEvents,
                         bool CallerNeedsEvent) {
#if XPTI_ENABLE_INSTRUMENTATION
  // We need a code pointer value and we use the object ptr; if code location
  // information is available, we will have function name and source file
  // information
  const char *UserData = "memory_transfer_node::memset", *FuncName = nullptr;
  // We have to get the stashed code location when not available
  detail::tls_code_loc_t Tls;
  auto CodeLocation = Tls.query();
  if (!CodeLocation.functionName())
    // If the code location is not available, we use the user data
    FuncName = UserData;
  else
    FuncName = CodeLocation.functionName();
  xpti::framework::tracepoint_scope_t TP(
      CodeLocation.fileName(), FuncName, CodeLocation.lineNumber(),
      CodeLocation.columnNumber(), (void *)this);
  TP.stream(detail::getActiveXPTIStreamID())
      .traceType(xpti::trace_point_type_t::node_create)
      .parentEvent(detail::GSYCLGraphEvent);

  // This information is necessary for memset, so we will not guard it by debug
  // stream check.
  TP.addMetadata([&](auto TEvent) {
    xpti::addMetadata(TEvent, "sycl_device",
                      reinterpret_cast<size_t>(MDevice.getHandleRef()));
    xpti::addMetadata(TEvent, "memory_ptr", reinterpret_cast<size_t>(Ptr));
    xpti::addMetadata(TEvent, "value_set", Value);
    xpti::addMetadata(TEvent, "memory_size", Count);
    xpti::addMetadata(TEvent, "queue_id", MQueueID);
  });

  // Before we notifiy the subscribers, we broadcast the 'queue_id', which was a
  // metadata entry to TLS for use by callback handlers
  xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, MQueueID);
  // Notify XPTI about the memset submission, which will create a memory object
  // node
  TP.notify(UserData);
  // Emit a begin/end scope for this call
  TP.scopedNotify((uint16_t)xpti::trace_point_type_t::task_begin, UserData);
#endif
  const std::vector<unsigned char> Pattern{static_cast<unsigned char>(Value)};
  return submitMemOpHelper(
      DepEvents, CallerNeedsEvent,
      [&](handler &CGH) { CGH.memset(Ptr, Value, Count); },
      MemoryManager::fill_usm, Ptr, *this, Count, Pattern);
}

void report(const code_location &CodeLoc) {
  std::cout << "Exception caught at ";
  if (CodeLoc.fileName())
    std::cout << "File: " << CodeLoc.fileName();
  if (CodeLoc.functionName())
    std::cout << " | Function: " << CodeLoc.functionName();
  if (CodeLoc.lineNumber())
    std::cout << " | Line: " << CodeLoc.lineNumber();
  if (CodeLoc.columnNumber())
    std::cout << " | Column: " << CodeLoc.columnNumber();
  std::cout << '\n';
}

event queue_impl::memcpy(void *Dest, const void *Src, size_t Count,
                         const std::vector<event> &DepEvents,
                         bool CallerNeedsEvent, const code_location &CodeLoc) {
#if XPTI_ENABLE_INSTRUMENTATION
  // We need a code pointer value and we duse the object ptr; If code location
  // is available, we use the source file information along with the object
  // pointer.
  xpti::framework::tracepoint_scope_t TP(
      CodeLoc.fileName(), CodeLoc.functionName(), CodeLoc.lineNumber(),
      CodeLoc.columnNumber(), (void *)this);
  TP.stream(detail::getActiveXPTIStreamID())
      .traceType(xpti::trace_point_type_t::node_create)
      .parentEvent(GSYCLGraphEvent);
  const char *UserData = "memory_transfer_node::memcpy";
  // We will include this metadata information as it is required for memcpy.
  TP.addMetadata([&](auto TEvent) {
    xpti::addMetadata(TEvent, "sycl_device",
                      reinterpret_cast<size_t>(MDevice.getHandleRef()));
    xpti::addMetadata(TEvent, "src_memory_ptr", reinterpret_cast<size_t>(Src));
    xpti::addMetadata(TEvent, "dest_memory_ptr",
                      reinterpret_cast<size_t>(Dest));
    xpti::addMetadata(TEvent, "memory_size", Count);
    xpti::addMetadata(TEvent, "queue_id", MQueueID);
  });
  // Before we notify the subscribers, we stash the 'queue_id', which was a
  // metadata entry to TLS for use by callback handlers
  xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, MQueueID);
  // Notify XPTI about the memcpy submission
  TP.notify(UserData);
  // Emit a begin/end scope for this call
  TP.scopedNotify((uint16_t)xpti::trace_point_type_t::task_begin, UserData);
#endif

  if ((!Src || !Dest) && Count != 0) {
    report(CodeLoc);
    throw exception(make_error_code(errc::invalid),
                    "NULL pointer argument in memory copy operation.");
  }
  return submitMemOpHelper(
      DepEvents, CallerNeedsEvent,
      [&](handler &CGH) { CGH.memcpy(Dest, Src, Count); },
      MemoryManager::copy_usm, Src, *this, Count, Dest);
}

event queue_impl::mem_advise(const void *Ptr, size_t Length,
                             ur_usm_advice_flags_t Advice,
                             const std::vector<event> &DepEvents,
                             bool CallerNeedsEvent) {
  return submitMemOpHelper(
      DepEvents, CallerNeedsEvent,
      [&](handler &CGH) { CGH.mem_advise(Ptr, Length, Advice); },
      MemoryManager::advise_usm, Ptr, *this, Length, Advice);
}

event queue_impl::memcpyToDeviceGlobal(void *DeviceGlobalPtr, const void *Src,
                                       bool IsDeviceImageScope, size_t NumBytes,
                                       size_t Offset,
                                       const std::vector<event> &DepEvents,
                                       bool CallerNeedsEvent) {
  return submitMemOpHelper(
      DepEvents, CallerNeedsEvent,
      [&](handler &CGH) {
        CGH.memcpyToDeviceGlobal(DeviceGlobalPtr, Src, IsDeviceImageScope,
                                 NumBytes, Offset);
      },
      MemoryManager::copy_to_device_global, DeviceGlobalPtr, IsDeviceImageScope,
      *this, NumBytes, Offset, Src);
}

event queue_impl::memcpyFromDeviceGlobal(void *Dest,
                                         const void *DeviceGlobalPtr,
                                         bool IsDeviceImageScope,
                                         size_t NumBytes, size_t Offset,
                                         const std::vector<event> &DepEvents,
                                         bool CallerNeedsEvent) {
  return submitMemOpHelper(
      DepEvents, CallerNeedsEvent,
      [&](handler &CGH) {
        CGH.memcpyFromDeviceGlobal(Dest, DeviceGlobalPtr, IsDeviceImageScope,
                                   NumBytes, Offset);
      },
      MemoryManager::copy_from_device_global, DeviceGlobalPtr,
      IsDeviceImageScope, *this, NumBytes, Offset, Dest);
}

sycl::detail::optional<event> queue_impl::getLastEvent() {
  // The external event is required to finish last if set, so it is considered
  // the last event if present.
  if (std::optional<event> ExternalEvent = MInOrderExternalEvent.read())
    return ExternalEvent;

  std::lock_guard<std::mutex> Lock{MMutex};
  if (MEmpty)
    return std::nullopt;
  auto &LastEvent = MGraph.expired() ? MDefaultGraphDeps.LastEventPtr
                                     : MExtGraphDeps.LastEventPtr;
  // If the event comes from a graph, we must return it.
  if (LastEvent)
    return detail::createSyclObjFromImpl<event>(LastEvent);
  // We insert a marker to represent an event at end.
  return detail::createSyclObjFromImpl<event>(insertMarkerEvent());
}

void queue_impl::addEvent(const detail::EventImplPtr &EventImpl) {
  if (!EventImpl)
    return;
  Command *Cmd = EventImpl->getCommand();
  if (Cmd != nullptr && EventImpl->getHandle() == nullptr) {
    std::weak_ptr<event_impl> EventWeakPtr{EventImpl};
    std::lock_guard<std::mutex> Lock{MMutex};
    MEventsWeak.push_back(std::move(EventWeakPtr));
  }
}

detail::EventImplPtr
queue_impl::submit_impl(const detail::type_erased_cgfo_ty &CGF,
                        bool CallerNeedsEvent, const detail::code_location &Loc,
                        bool IsTopCodeLoc,
                        const v1::SubmissionInfo &SubmitInfo) {
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  detail::handler_impl HandlerImplVal(*this, CallerNeedsEvent);
  handler Handler(HandlerImplVal);
#else
  handler Handler(shared_from_this(), CallerNeedsEvent);
#endif

#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (xptiTraceEnabled()) {
    Handler.saveCodeLoc(Loc, IsTopCodeLoc);
  }
#endif

  {
    NestedCallsTracker tracker;
    CGF(Handler);
  }

  // We might swap handlers as part of the CGH(Handler) call in the reduction
  // case, so need to retrieve the handler_impl reference after that.
  detail::handler_impl &HandlerImpl = *detail::getSyclObjImpl(Handler);

  // Scheduler will later omit events, that are not required to execute tasks.
  // Host and interop tasks, however, are not submitted to low-level runtimes
  // and require separate dependency management.
  const CGType Type = HandlerImpl.MCGType;
  std::vector<StreamImplPtr> Streams;
  if (Type == CGType::Kernel)
    Streams = std::move(Handler.MStreamStorage);

  HandlerImpl.MEventMode = SubmitInfo.EventMode();

  auto isHostTask = Type == CGType::CodeplayHostTask ||
                    (Type == CGType::ExecCommandBuffer &&
                     HandlerImpl.MExecGraph->containsHostTask());

  auto noLastEventPath = !isHostTask &&
                         MNoLastEventMode.load(std::memory_order_acquire) &&
                         !Streams.size();

  if (noLastEventPath) {
    std::unique_lock<std::mutex> Lock(MMutex);

    // Check if we are still in no last event mode. There could
    // have been a concurrent submit.
    if (MNoLastEventMode.load(std::memory_order_relaxed)) {
      return finalizeHandlerInOrderNoEventsUnlocked(Handler);
    }
  }

  detail::EventImplPtr EventImpl;
  if (!isInOrder()) {
    EventImpl = finalizeHandlerOutOfOrder(Handler);
    addEvent(EventImpl);
  } else {
    if (isHostTask) {
      std::unique_lock<std::mutex> Lock(MMutex);
      EventImpl = finalizeHandlerInOrderHostTaskUnlocked(Handler);
    } else {
      std::unique_lock<std::mutex> Lock(MMutex);

      if (trySwitchingToNoEventsMode()) {
        EventImpl = finalizeHandlerInOrderNoEventsUnlocked(Handler);
      } else {
        EventImpl = finalizeHandlerInOrderWithDepsUnlocked(Handler);
      }
    }
  }

  for (auto &Stream : Streams) {
    // We don't want stream flushing to be blocking operation that is why submit
    // a host task to print stream buffer. It will fire up as soon as the kernel
    // finishes execution.
    auto L = [&](handler &ServiceCGH) {
      Stream->generateFlushCommand(ServiceCGH);
    };
    detail::type_erased_cgfo_ty CGF{L};
    detail::EventImplPtr FlushEvent =
        submit_impl(CGF, /*CallerNeedsEvent*/ true, Loc, IsTopCodeLoc, {});
    if (EventImpl)
      EventImpl->attachEventToCompleteWeak(FlushEvent);
    if (!isInOrder()) {
      // For in-order queue, the dependencies will be tracked by LastEvent
      registerStreamServiceEvent(FlushEvent);
    }
  }

  return EventImpl;
}

EventImplPtr queue_impl::submit_kernel_scheduler_bypass(
    KernelData &KData, std::vector<detail::EventImplPtr> &DepEvents,
    bool EventNeeded, detail::kernel_impl *KernelImplPtr,
    detail::kernel_bundle_impl *KernelBundleImpPtr,
    const detail::code_location &CodeLoc, bool IsTopCodeLoc) {
  std::vector<ur_event_handle_t> RawEvents;

  // TODO checking the size of the events vector and avoiding the call is
  // more efficient here at this point
  if (DepEvents.size() > 0) {
    RawEvents = detail::Command::getUrEvents(DepEvents, this, false);
  }

  bool DiscardEvent = !EventNeeded && supportsDiscardingPiEvents();
  std::shared_ptr<detail::event_impl> ResultEvent =
      DiscardEvent ? nullptr : detail::event_impl::create_device_event(*this);

  auto EnqueueKernel = [&]() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    xpti_td *CmdTraceEvent = nullptr;
    uint64_t InstanceID = 0;
    uint8_t StreamID = 0;
    // Only enable instrumentation if there are subscribes to the SYCL
    // stream
    const bool xptiEnabled = xptiTraceEnabled();
    if (xptiEnabled) {
      StreamID = detail::getActiveXPTIStreamID();
      std::tie(CmdTraceEvent, InstanceID) = emitKernelInstrumentationData(
          StreamID, KernelImplPtr, CodeLoc, IsTopCodeLoc,
          *KData.getDeviceKernelInfoPtr(), this, KData.getNDRDesc(),
          KernelBundleImpPtr, KData.getArgs());
      detail::emitInstrumentationGeneral(StreamID, InstanceID, CmdTraceEvent,
                                         xpti::trace_task_begin, nullptr);
    }
#endif
    const detail::RTDeviceBinaryImage *BinImage = nullptr;
    if (detail::SYCLConfig<detail::SYCL_JIT_AMDGCN_PTX_KERNELS>::get()) {
      BinImage = detail::retrieveKernelBinary(*this, KData.getKernelName());
      assert(BinImage && "Failed to obtain a binary image.");
    }
    enqueueImpKernel(*this, KData.getNDRDesc(), KData.getArgs(),
                     KernelBundleImpPtr, KernelImplPtr,
                     *KData.getDeviceKernelInfoPtr(), RawEvents,
                     ResultEvent.get(), nullptr, KData.getKernelCacheConfig(),
                     KData.isCooperative(), KData.usesClusterLaunch(),
                     KData.getKernelWorkGroupMemorySize(), BinImage,
                     KData.getKernelFuncPtr());
#ifdef XPTI_ENABLE_INSTRUMENTATION
    if (xptiEnabled) {
      // Emit signal only when event is created
      if (!DiscardEvent) {
        detail::emitInstrumentationGeneral(
            StreamID, InstanceID, CmdTraceEvent, xpti::trace_signal,
            static_cast<const void *>(ResultEvent->getHandle()));
      }
      detail::emitInstrumentationGeneral(StreamID, InstanceID, CmdTraceEvent,
                                         xpti::trace_task_end, nullptr);
    }
#endif
  };

  if (DiscardEvent) {
    EnqueueKernel();
  } else {
    ResultEvent->setWorkerQueue(weak_from_this());
    ResultEvent->setStateIncomplete();
    ResultEvent->setSubmissionTime();

    EnqueueKernel();
    ResultEvent->setEnqueued();
    // connect returned event with dependent events
    if (!isInOrder()) {
      // DepEvents is not used anymore, so can move.
      ResultEvent->getPreparedDepsEvents() = std::move(DepEvents);
      // ResultEvent is local for current thread, no need to lock.
      ResultEvent->cleanDepEventsThroughOneLevelUnlocked();
    }
  }

  return ResultEvent;
}

EventImplPtr queue_impl::submit_command_to_graph(
    ext::oneapi::experimental::detail::graph_impl &GraphImpl,
    std::unique_ptr<detail::CG> CommandGroup, sycl::detail::CGType CGType,
    sycl::ext::oneapi::experimental::node_type UserFacingNodeType) {
  auto EventImpl = detail::event_impl::create_completed_host_event();
  EventImpl->setSubmittedQueue(this);
  ext::oneapi::experimental::detail::node_impl *NodeImpl = nullptr;

  // GraphImpl is read and written in this scope so we lock this graph
  // with full priviledges.
  ext::oneapi::experimental::detail::graph_impl::WriteLock Lock(
      GraphImpl.MMutex);

  ext::oneapi::experimental::node_type NodeType =
      UserFacingNodeType != ext::oneapi::experimental::node_type::empty
          ? UserFacingNodeType
          : ext::oneapi::experimental::detail::getNodeTypeFromCG(CGType);

  // Create a new node in the graph representing this command-group
  if (isInOrder()) {
    // In-order queues create implicit linear dependencies between nodes.
    // Find the last node added to the graph from this queue, so our new
    // node can set it as a predecessor.
    std::vector<ext::oneapi::experimental::detail::node_impl *> Deps;
    if (ext::oneapi::experimental::detail::node_impl *DependentNode =
            GraphImpl.getLastInorderNode(this)) {
      Deps.push_back(DependentNode);
    }
    NodeImpl = &GraphImpl.add(NodeType, std::move(CommandGroup), Deps);

    // If we are recording an in-order queue remember the new node, so it
    // can be used as a dependency for any more nodes recorded from this
    // queue.
    GraphImpl.setLastInorderNode(*this, *NodeImpl);
  } else {
    ext::oneapi::experimental::detail::node_impl *LastBarrierRecordedFromQueue =
        GraphImpl.getBarrierDep(weak_from_this());
    std::vector<ext::oneapi::experimental::detail::node_impl *> Deps;

    if (LastBarrierRecordedFromQueue) {
      Deps.push_back(LastBarrierRecordedFromQueue);
    }
    NodeImpl = &GraphImpl.add(NodeType, std::move(CommandGroup), Deps);

    if (NodeImpl->MCGType == sycl::detail::CGType::Barrier) {
      GraphImpl.setBarrierDep(weak_from_this(), *NodeImpl);
    }
  }

  // Associate an event with this new node and return the event.
  GraphImpl.addEventForNode(EventImpl, *NodeImpl);

  return EventImpl;
}

EventImplPtr queue_impl::submit_kernel_direct_impl(
    const NDRDescT &NDRDesc, detail::HostKernelRefBase &HostKernel,
    detail::DeviceKernelInfo *DeviceKernelInfo, bool CallerNeedsEvent,
    sycl::span<const event> DepEvents,
    const detail::KernelPropertyHolderStructTy &Props,
    const detail::code_location &CodeLoc, bool IsTopCodeLoc) {

  KernelData KData;

  KData.setDeviceKernelInfoPtr(DeviceKernelInfo);
  KData.setNDRDesc(NDRDesc);

  // Validate and set kernel launch properties.
  KData.validateAndSetKernelLaunchProperties(Props, hasCommandGraph(),
                                             getDeviceImpl());

  auto SubmitKernelFunc = [&](detail::CG::StorageInitHelper &&CGData,
                              bool SchedulerBypass) -> EventImplPtr {
    if (SchedulerBypass) {
      // No need to copy/move the kernel function, so we set
      // the function pointer to the original function
      KData.setKernelFunc(HostKernel.getPtr());

      return submit_kernel_scheduler_bypass(KData, CGData.MEvents,
                                            CallerNeedsEvent, nullptr, nullptr,
                                            CodeLoc, IsTopCodeLoc);
    }
    std::unique_ptr<detail::CG> CommandGroup;
    std::vector<std::shared_ptr<detail::stream_impl>> StreamStorage;
    std::vector<std::shared_ptr<const void>> AuxiliaryResources;

    std::shared_ptr<detail::HostKernelBase> HostKernelPtr =
        HostKernel.takeOrCopyOwnership();

    // When the kernel function is stored for future use,
    // set the function pointer to the stored function
    KData.setKernelFunc(HostKernelPtr->getPtr());

    KData.extractArgsAndReqsFromLambda();

    CommandGroup.reset(new detail::CGExecKernel(
        KData.getNDRDesc(), std::move(HostKernelPtr),
        nullptr, // Kernel
        nullptr, // KernelBundle
        std::move(CGData), std::move(KData).getArgs(),
        *KData.getDeviceKernelInfoPtr(), std::move(StreamStorage),
        std::move(AuxiliaryResources), detail::CGType::Kernel,
        KData.getKernelCacheConfig(), KData.isCooperative(),
        KData.usesClusterLaunch(), KData.getKernelWorkGroupMemorySize(),
        CodeLoc));
    CommandGroup->MIsTopCodeLoc = IsTopCodeLoc;

    if (auto GraphImpl = getCommandGraph(); GraphImpl) {
      return submit_command_to_graph(*GraphImpl, std::move(CommandGroup),
                                     detail::CGType::Kernel);
    }

    return detail::Scheduler::getInstance().addCG(std::move(CommandGroup),
                                                  *this, true);
  };

  return submit_direct(CallerNeedsEvent, DepEvents, SubmitKernelFunc);
}

template <typename SubmitCommandFuncType>
detail::EventImplPtr
queue_impl::submit_direct(bool CallerNeedsEvent,
                          sycl::span<const event> DepEvents,
                          SubmitCommandFuncType &SubmitCommandFunc) {
  detail::CG::StorageInitHelper CGData;
  std::unique_lock<std::mutex> Lock(MMutex);

  // Used by queue_empty() and getLastEvent()
  MEmpty.store(false, std::memory_order_release);

  // Sync with an external event
  std::optional<event> ExternalEvent = popExternalEvent();
  if (ExternalEvent) {
    registerEventDependency</*LockQueue*/ false>(
        getSyclObjImpl(*ExternalEvent), CGData.MEvents, this, getContextImpl(),
        getDeviceImpl(), hasCommandGraph() ? getCommandGraph().get() : nullptr,
        detail::CGType::Kernel);
  }

  auto &Deps = hasCommandGraph() ? MExtGraphDeps : MDefaultGraphDeps;

  // Sync with the last event for in order queue
  EventImplPtr &LastEvent = Deps.LastEventPtr;
  if (isInOrder() && LastEvent) {
    registerEventDependency</*LockQueue*/ false>(
        LastEvent, CGData.MEvents, this, getContextImpl(), getDeviceImpl(),
        hasCommandGraph() ? getCommandGraph().get() : nullptr,
        detail::CGType::Kernel);
  }

  for (event e : DepEvents) {
    registerEventDependency</*LockQueue*/ false>(
        getSyclObjImpl(e), CGData.MEvents, this, getContextImpl(),
        getDeviceImpl(), hasCommandGraph() ? getCommandGraph().get() : nullptr,
        detail::CGType::Kernel);
  }

  // Barrier and un-enqueued commands synchronization for out or order queue
  if (!isInOrder()) {
    MMissedCleanupRequests.unset(
        [&](MissedCleanupRequestsType &MissedCleanupRequests) {
          for (auto &UpdatedGraph : MissedCleanupRequests)
            doUnenqueuedCommandCleanup(UpdatedGraph);
          MissedCleanupRequests.clear();
        });

    if (Deps.LastBarrier && !Deps.LastBarrier->isEnqueued()) {
      CGData.MEvents.push_back(Deps.LastBarrier);
    }
  }

  bool SchedulerBypass =
      (CGData.MEvents.size() > 0
           ? detail::Scheduler::areEventsSafeForSchedulerBypass(
                 CGData.MEvents, getContextImpl())
           : true) &&
      !hasCommandGraph();

  // Synchronize with the "no last event mode", used by the handler-based
  // kernel submit path
  MNoLastEventMode.store(isInOrder() && SchedulerBypass,
                         std::memory_order_relaxed);

  EventImplPtr EventImpl =
      SubmitCommandFunc(std::move(CGData), SchedulerBypass);

  // Sync with the last event for in order queue. For scheduler-bypass flow,
  // the ordering is done at the layers below the SYCL runtime,
  // but for the scheduler-based flow, it needs to be done here, as the
  // scheduler handles host task submissions.
  if (isInOrder()) {
    LastEvent = SchedulerBypass ? nullptr : EventImpl;
  }

  // Barrier and un-enqueued commands synchronization for out or order queue
  if (!isInOrder() && !EventImpl->isEnqueued()) {
    Deps.UnenqueuedCmdEvents.push_back(EventImpl);
  }

  return CallerNeedsEvent ? std::move(EventImpl) : nullptr;
}

template <typename HandlerFuncT>
event queue_impl::submitWithHandler(const std::vector<event> &DepEvents,
                                    bool CallerNeedsEvent,
                                    HandlerFuncT HandlerFunc) {
  v1::SubmissionInfo SI{};
  auto L = [&](handler &CGH) {
    CGH.depends_on(DepEvents);
    HandlerFunc(CGH);
  };
  detail::type_erased_cgfo_ty CGF{L};

  if (!CallerNeedsEvent && supportsDiscardingPiEvents()) {
    submit_without_event(CGF, SI,
                         /*CodeLoc*/ {}, /*IsTopCodeLoc*/ true);
    return createSyclObjFromImpl<event>(event_impl::create_discarded_event());
  }
  return submit_with_event(CGF, SI,
                           /*CodeLoc*/ {}, /*IsTopCodeLoc*/ true);
}

template <typename HandlerFuncT, typename MemOpFuncT, typename... MemOpArgTs>
event queue_impl::submitMemOpHelper(const std::vector<event> &DepEvents,
                                    bool CallerNeedsEvent,
                                    HandlerFuncT HandlerFunc,
                                    MemOpFuncT MemOpFunc,
                                    MemOpArgTs &&...MemOpArgs) {
  // We need to submit command and update the last event under same lock if we
  // have in-order queue.
  {
    std::unique_lock<std::mutex> Lock(MMutex, std::defer_lock);

    std::vector<event> MutableDepEvents;
    const std::vector<event> &ExpandedDepEvents =
        getExtendDependencyList(DepEvents, MutableDepEvents, Lock);

    MEmpty = false;

    // If we have a command graph set we need to capture the op through the
    // handler rather than by-passing the scheduler.
    if (MGraph.expired() && Scheduler::areEventsSafeForSchedulerBypass(
                                ExpandedDepEvents, *MContext)) {
      auto isNoEventsMode = trySwitchingToNoEventsMode();
      if (!CallerNeedsEvent && isNoEventsMode) {
        NestedCallsTracker tracker;
        MemOpFunc(std::forward<MemOpArgTs>(MemOpArgs)...,
                  getUrEvents(ExpandedDepEvents),
                  /*PiEvent*/ nullptr);

        return createSyclObjFromImpl<event>(
            event_impl::create_discarded_event());
      }

      event ResEvent = prepareSYCLEventAssociatedWithQueue(*this);
      const auto &EventImpl = detail::getSyclObjImpl(ResEvent);
      {
        NestedCallsTracker tracker;
        ur_event_handle_t UREvent = nullptr;
        EventImpl->setSubmissionTime();
        MemOpFunc(std::forward<MemOpArgTs>(MemOpArgs)...,
                  getUrEvents(ExpandedDepEvents), &UREvent);
        EventImpl->setHandle(UREvent);
        EventImpl->setEnqueued();
        // connect returned event with dependent events
        if (!isInOrder()) {
          std::vector<EventImplPtr> &ExpandedDepEventImplPtrs =
              EventImpl->getPreparedDepsEvents();
          ExpandedDepEventImplPtrs.reserve(ExpandedDepEvents.size());
          for (const event &DepEvent : ExpandedDepEvents)
            ExpandedDepEventImplPtrs.push_back(
                detail::getSyclObjImpl(DepEvent));

          // EventImpl is local for current thread, no need to lock.
          EventImpl->cleanDepEventsThroughOneLevelUnlocked();
        }
      }

      if (isInOrder() && !isNoEventsMode) {
        auto &EventToStoreIn = MGraph.expired() ? MDefaultGraphDeps.LastEventPtr
                                                : MExtGraphDeps.LastEventPtr;
        EventToStoreIn = EventImpl;
      }

      return ResEvent;
    }
  }
  return submitWithHandler(DepEvents, CallerNeedsEvent, HandlerFunc);
}

#ifdef XPTI_ENABLE_INSTRUMENTATION
void *queue_impl::instrumentationProlog(const detail::code_location &CodeLoc,
                                        std::string &Name,
                                        xpti::stream_id_t StreamID,
                                        uint64_t &IId) {
  void *TraceEvent = nullptr;
  constexpr uint16_t NotificationTraceType = xpti::trace_wait_begin;
  if (!xptiCheckTraceEnabled(StreamID, NotificationTraceType))
    return TraceEvent;

  xpti_tracepoint_t *Event;
  // We try to create a unique string for the wait() call by combining it with
  // the queue address
  xpti::utils::StringHelper NG;
  Name = NG.nameWithAddress<queue_impl *>("queue.wait", this);

  bool HasSourceInfo = CodeLoc.fileName() != nullptr;
  // wait() calls could be at different user-code locations; We create a new
  // event based on the code location info and if this has been seen before, a
  // previously created event will be returned.
  if (HasSourceInfo) {
    Event = xptiCreateTracepoint(CodeLoc.functionName(), CodeLoc.fileName(),
                                 CodeLoc.lineNumber(), CodeLoc.columnNumber(),
                                 (void *)this);
  } else {
    Event = xptiCreateTracepoint(Name.c_str(), nullptr, 0, 0, (void *)this);
  }

  IId = xptiGetUniqueId();
  auto WaitEvent = Event->event_ref();
  // We will allow the device type to be set
  xpti::addMetadata(WaitEvent, "sycl_device_type", queueDeviceToString(this));
  // We limit the amount of metadata that is added to the regular stream.
  // Only "sycl.debug" stream will have the full information. This improves the
  // performance when this data is not required by the tool or the collector.
  if (isDebugStream(StreamID)) {
    if (HasSourceInfo) {
      xpti::addMetadata(WaitEvent, "sym_function_name", CodeLoc.functionName());
      xpti::addMetadata(WaitEvent, "sym_source_file_name", CodeLoc.fileName());
      xpti::addMetadata(WaitEvent, "sym_line_no",
                        static_cast<xpti::object_id_t>((CodeLoc.lineNumber())));
      xpti::addMetadata(
          WaitEvent, "sym_column_no",
          static_cast<xpti::object_id_t>((CodeLoc.columnNumber())));
    }
  }
  xptiNotifySubscribers(StreamID, xpti::trace_wait_begin, nullptr, WaitEvent,
                        IId, static_cast<const void *>(Name.c_str()));
  TraceEvent = (void *)WaitEvent;

  return TraceEvent;
}

void queue_impl::instrumentationEpilog(void *TelemetryEvent, std::string &Name,
                                       xpti::stream_id_t StreamID,
                                       uint64_t IId) {
  constexpr uint16_t NotificationTraceType = xpti::trace_wait_end;
  if (!(xptiCheckTraceEnabled(StreamID, NotificationTraceType) &&
        TelemetryEvent))
    return;
  // Close the wait() scope
  xpti::trace_event_data_t *TraceEvent =
      (xpti::trace_event_data_t *)TelemetryEvent;
  xptiNotifySubscribers(StreamID, NotificationTraceType, nullptr, TraceEvent,
                        IId, static_cast<const void *>(Name.c_str()));
}

#endif // XPTI_ENABLE_INSTRUMENTATION

void queue_impl::wait(const detail::code_location &CodeLoc) {
  (void)CodeLoc;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  void *TelemetryEvent = nullptr;
  uint64_t IId;
  std::string Name;
  uint8_t StreamID = 0;
  const bool xptiEnabled = xptiTraceEnabled();
  if (xptiEnabled) {
    StreamID = detail::getActiveXPTIStreamID();
    TelemetryEvent = instrumentationProlog(CodeLoc, Name, StreamID, IId);
  }
#endif

  if (!MGraph.expired()) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "wait cannot be called for a queue which is "
                          "recording to a command graph.");
  }

  // If there is an external event set, we know we are using an in-order queue
  // and the event is required to finish after the last event in the queue. As
  // such, we can just wait for it and finish.
  std::optional<event> ExternalEvent = popExternalEvent();
  if (ExternalEvent) {
    ExternalEvent->wait();

    // Additionally, we can clean up the event lists that we would have
    // otherwise cleared.
    if (!MEventsWeak.empty()) {
      std::lock_guard<std::mutex> Lock(MMutex);
      MEventsWeak.clear();
    }
    if (!MStreamsServiceEvents.empty()) {
      std::lock_guard<std::mutex> Lock(MStreamsServiceEventsMutex);
      MStreamsServiceEvents.clear();
    }
  }

  if (isInOrder() && !MNoLastEventMode.load(std::memory_order_relaxed)) {
    // if MLastEvent is not null, we need to wait for it
    EventImplPtr LastEvent;
    {
      std::lock_guard<std::mutex> Lock(MMutex);
      LastEvent = MDefaultGraphDeps.LastEventPtr;
    }
    if (LastEvent) {
      LastEvent->wait();
    }
  } else if (!isInOrder()) {
    std::vector<std::weak_ptr<event_impl>> WeakEvents;
    {
      std::lock_guard<std::mutex> Lock(MMutex);
      WeakEvents.swap(MEventsWeak);
      MMissedCleanupRequests.unset(
          [&](MissedCleanupRequestsType &MissedCleanupRequests) {
            for (auto &UpdatedGraph : MissedCleanupRequests)
              doUnenqueuedCommandCleanup(UpdatedGraph);
            MissedCleanupRequests.clear();
          });
    }

    // Wait for unenqueued or host task events, starting
    // from the latest submitted task in order to minimize total amount of
    // calls, then handle the rest with urQueueFinish.
    for (auto EventImplWeakPtrIt = WeakEvents.rbegin();
         EventImplWeakPtrIt != WeakEvents.rend(); ++EventImplWeakPtrIt) {
      if (std::shared_ptr<event_impl> EventImplSharedPtr =
              EventImplWeakPtrIt->lock()) {
        // A nullptr UR event indicates that urQueueFinish will not cover it,
        // either because it's a host task event or an unenqueued one.
        if (nullptr == EventImplSharedPtr->getHandle()) {
          EventImplSharedPtr->wait();
        }
      }
    }
  }

  getAdapter().call<UrApiKind::urQueueFinish>(getHandleRef());

  if (!isInOrder()) {
    std::vector<EventImplPtr> StreamsServiceEvents;
    {
      std::lock_guard<std::mutex> Lock(MStreamsServiceEventsMutex);
      StreamsServiceEvents.swap(MStreamsServiceEvents);
    }
    for (const EventImplPtr &Event : StreamsServiceEvents)
      Event->wait();
  }

#ifdef XPTI_ENABLE_INSTRUMENTATION
  // There is an early return in instrumentationEpilog() if no subscribers are
  // subscribing to queue.wait().
  if (xptiEnabled) {
    instrumentationEpilog(TelemetryEvent, Name, StreamID, IId);
  }
#endif
}

void queue_impl::constructorNotification() {
#if XPTI_ENABLE_INSTRUMENTATION
  // If there are no subscribers to queue_create, return immediately.
  constexpr uint16_t NotificationTraceType =
      static_cast<uint16_t>(xpti::trace_point_type_t::queue_create);
  if (!anyTraceEnabled(NotificationTraceType))
    return;
  // We do not have CodeLoc for the queue constructor, so we will have to create
  // a queue name with the queue ID to create an event; this step can be avoided
  // by using CodeLoc.
  xpti::utils::StringHelper SH;
  std::string AddrStr = SH.addressAsString<size_t>(MQueueID);
  std::string QueueName = SH.nameWithAddressString("queue", AddrStr);

  xpti_tracepoint_t *Event =
      xptiCreateTracepoint(QueueName.c_str(), nullptr, 0, 0, (void *)this);
  MInstanceID = xptiGetUniqueId();
  xpti_td *TEvent = Event->event_ref();
  // Cache the trace event, stream id and instance IDs for the destructor.
  MTraceEvent = (void *)TEvent;
  // We will allow the queue metadata to be set as this is performed
  // infrequently.
  xpti::addMetadata(TEvent, "sycl_context",
                    reinterpret_cast<size_t>(MContext->getHandleRef()));
  xpti::addMetadata(TEvent, "sycl_device_name",
                    MDevice.get_info<info::device::name>());
  xpti::addMetadata(TEvent, "sycl_device",
                    reinterpret_cast<size_t>(MDevice.getHandleRef()));
  xpti::addMetadata(TEvent, "is_inorder", MIsInorder);
  xpti::addMetadata(TEvent, "queue_id", MQueueID);
  xpti::addMetadata(TEvent, "queue_handle",
                    reinterpret_cast<size_t>(getHandleRef()));
  // Also publish to TLS before notification.
  xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, MQueueID);
  xptiNotifySubscribers(detail::getActiveXPTIStreamID(),
                        (uint16_t)xpti::trace_point_type_t::queue_create,
                        nullptr, TEvent, MInstanceID,
                        static_cast<const void *>("queue_create"));
#endif
}

void queue_impl::destructorNotification() {
#if XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      static_cast<uint16_t>(xpti::trace_point_type_t::queue_destroy);
  if (anyTraceEnabled(NotificationTraceType)) {
    // Use the cached trace event, stream id and instance IDs for the
    // destructor
    xptiNotifySubscribers(detail::getActiveXPTIStreamID(),
                          NotificationTraceType, nullptr,
                          (xpti::trace_event_data_t *)MTraceEvent, MInstanceID,
                          static_cast<const void *>("queue_destroy"));
    xptiReleaseEvent((xpti::trace_event_data_t *)MTraceEvent);
  }
#endif
}

ur_native_handle_t queue_impl::getNative(int32_t &NativeHandleDesc) const {
  ur_native_handle_t Handle{};
  ur_queue_native_desc_t UrNativeDesc{UR_STRUCTURE_TYPE_QUEUE_NATIVE_DESC,
                                      nullptr, nullptr};
  UrNativeDesc.pNativeData = &NativeHandleDesc;

  getAdapter().call<UrApiKind::urQueueGetNativeHandle>(MQueue, &UrNativeDesc,
                                                       &Handle);
  if (getContextImpl().getBackend() == backend::opencl)
    __SYCL_OCL_CALL(clRetainCommandQueue, ur::cast<cl_command_queue>(Handle));

  return Handle;
}

bool queue_impl::queue_empty() const {
  // If we have in-order queue with non-empty last event, just check its status.
  if (isInOrder()) {
    if (MEmpty.load(std::memory_order_acquire))
      return true;

    std::lock_guard<std::mutex> Lock(MMutex);

    if (MDefaultGraphDeps.LastEventPtr &&
        !MDefaultGraphDeps.LastEventPtr->isDiscarded())
      return MDefaultGraphDeps.LastEventPtr
                 ->get_info<info::event::command_execution_status>() ==
             info::event_command_status::complete;
  }

  // Check the status of the backend queue if this is not a host queue.
  ur_bool_t IsReady = false;
  getAdapter().call<UrApiKind::urQueueGetInfo>(
      MQueue, UR_QUEUE_INFO_EMPTY, sizeof(IsReady), &IsReady, nullptr);
  if (!IsReady)
    return false;

  // If got here, it means that LastEventPtr is nullptr (so no possible Host
  // Tasks) and there is nothing executing on the device.
  if (isInOrder())
    return true;

  // We may have events like host tasks which are not submitted to the backend
  // queue so we need to get their status separately.
  std::lock_guard<std::mutex> Lock(MMutex);
  for (auto EventImplWeakPtrIt = MEventsWeak.begin();
       EventImplWeakPtrIt != MEventsWeak.end(); ++EventImplWeakPtrIt)
    if (std::shared_ptr<event_impl> EventImplSharedPtr =
            EventImplWeakPtrIt->lock())
      if (EventImplSharedPtr->isHost() &&
          EventImplSharedPtr
                  ->get_info<info::event::command_execution_status>() !=
              info::event_command_status::complete)
        return false;

  // If we didn't exit early above then it means that all events in the queue
  // are completed.
  return true;
}

void queue_impl::revisitUnenqueuedCommandsState(
    const EventImplPtr &CompletedHostTask) {
  if (MIsInorder)
    return;
  std::unique_lock<std::mutex> Lock{MMutex, std::try_to_lock};
  if (Lock.owns_lock())
    doUnenqueuedCommandCleanup(CompletedHostTask->getCommandGraph());
  else {
    MMissedCleanupRequests.set(
        [&](MissedCleanupRequestsType &MissedCleanupRequests) {
          MissedCleanupRequests.push_back(CompletedHostTask->getCommandGraph());
        });
  }
}

void queue_impl::doUnenqueuedCommandCleanup(
    const std::shared_ptr<ext::oneapi::experimental::detail::graph_impl>
        &Graph) {
  auto tryToCleanup = [](DependencyTrackingItems &Deps) {
    if (Deps.LastBarrier && Deps.LastBarrier->isEnqueued()) {
      Deps.LastBarrier = nullptr;
      Deps.UnenqueuedCmdEvents.clear();
    } else {
      if (Deps.UnenqueuedCmdEvents.empty())
        return;
      Deps.UnenqueuedCmdEvents.erase(
          std::remove_if(
              Deps.UnenqueuedCmdEvents.begin(), Deps.UnenqueuedCmdEvents.end(),
              [](const EventImplPtr &CommandEvent) {
                return (CommandEvent->isHost() ? CommandEvent->isCompleted()
                                               : CommandEvent->isEnqueued());
              }),
          Deps.UnenqueuedCmdEvents.end());
    }
  };
  // Barrier enqueue could be significantly postponed due to host task
  // dependency if any. No guarantee that it will happen while same graph deps
  // are still recording.
  if (Graph && Graph == getCommandGraph())
    tryToCleanup(MExtGraphDeps);
  else
    tryToCleanup(MDefaultGraphDeps);
}

void queue_impl::verifyProps(const property_list &Props) const {
  auto CheckDataLessProperties = [](int PropertyKind) {
#define __SYCL_DATA_LESS_PROP_DEPRECATED_ALIAS(NS_QUALIFIER, PROP_NAME,        \
                                               ENUM_VAL, WARNING)              \
  case NS_QUALIFIER::PROP_NAME::getKind():                                     \
    return true;
#define __SYCL_DATA_LESS_PROP(NS_QUALIFIER, PROP_NAME, ENUM_VAL)               \
  case NS_QUALIFIER::PROP_NAME::getKind():                                     \
    return true;
    switch (PropertyKind) {
#include <sycl/properties/queue_properties.def>
    default:
      return false;
    }
  };
  auto CheckPropertiesWithData = [](int PropertyKind) {
#define __SYCL_MANUALLY_DEFINED_PROP(NS_QUALIFIER, PROP_NAME)                  \
  case NS_QUALIFIER::PROP_NAME::getKind():                                     \
    return true;
    switch (PropertyKind) {
#include <sycl/properties/queue_properties.def>
    default:
      return false;
    }
  };
  detail::PropertyValidator::checkPropsAndThrow(Props, CheckDataLessProperties,
                                                CheckPropertiesWithData);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
