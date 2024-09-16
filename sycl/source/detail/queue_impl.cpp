//==------------------ queue_impl.cpp - SYCL queue -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
#include "xpti/xpti_trace_framework.hpp"
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
    if (NestedCallsDetector)
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Calls to sycl::queue::submit cannot be nested. Command group "
          "function objects should use the sycl::handler API instead.");
    NestedCallsDetector = true;
  }

  ~NestedCallsTracker() { NestedCallsDetector = false; }
};

static std::vector<ur_event_handle_t>
getUrEvents(const std::vector<sycl::event> &DepEvents) {
  std::vector<ur_event_handle_t> RetUrEvents;
  for (const sycl::event &Event : DepEvents) {
    const EventImplPtr &EventImpl = detail::getSyclObjImpl(Event);
    auto Handle = EventImpl->getHandle();
    if (Handle != nullptr)
      RetUrEvents.push_back(Handle);
  }
  return RetUrEvents;
}

template <>
uint32_t queue_impl::get_info<info::queue::reference_count>() const {
  ur_result_t result = UR_RESULT_SUCCESS;
  getPlugin()->call<UrApiKind::urQueueGetInfo>(
      MQueues[0], UR_QUEUE_INFO_REFERENCE_COUNT, sizeof(result), &result,
      nullptr);
  return result;
}

template <> context queue_impl::get_info<info::queue::context>() const {
  return get_context();
}

template <> device queue_impl::get_info<info::queue::device>() const {
  return get_device();
}

template <>
typename info::platform::version::return_type
queue_impl::get_backend_info<info::platform::version>() const {
  if (getContextImplPtr()->getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::platform::version info descriptor can "
                          "only be queried with an OpenCL backend");
  }
  return get_device().get_platform().get_info<info::platform::version>();
}

template <>
typename info::device::version::return_type
queue_impl::get_backend_info<info::device::version>() const {
  if (getContextImplPtr()->getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::device::version info descriptor can only "
                          "be queried with an OpenCL backend");
  }
  return get_device().get_info<info::device::version>();
}

template <>
typename info::device::backend_version::return_type
queue_impl::get_backend_info<info::device::backend_version>() const {
  if (getContextImplPtr()->getBackend() != backend::ext_oneapi_level_zero) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::device::backend_version info descriptor "
                          "can only be queried with a Level Zero backend");
  }
  return "";
  // Currently The Level Zero backend does not define the value of this
  // information descriptor and implementations are encouraged to return the
  // empty string as per specification.
}

static event prepareSYCLEventAssociatedWithQueue(
    const std::shared_ptr<detail::queue_impl> &QueueImpl) {
  auto EventImpl = std::make_shared<detail::event_impl>(QueueImpl);
  EventImpl->setContextImpl(detail::getSyclObjImpl(QueueImpl->get_context()));
  EventImpl->setStateIncomplete();
  return detail::createSyclObjFromImpl<event>(EventImpl);
}

static event createDiscardedEvent() {
  EventImplPtr EventImpl =
      std::make_shared<event_impl>(event_impl::HES_Discarded);
  return createSyclObjFromImpl<event>(EventImpl);
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

event queue_impl::memset(const std::shared_ptr<detail::queue_impl> &Self,
                         void *Ptr, int Value, size_t Count,
                         const std::vector<event> &DepEvents,
                         bool CallerNeedsEvent) {
#if XPTI_ENABLE_INSTRUMENTATION
  // We need a code pointer value and we use the object ptr; if code location
  // information is available, we will have function name and source file
  // information
  XPTIScope PrepareNotify((void *)this,
                          (uint16_t)xpti::trace_point_type_t::node_create,
                          SYCL_STREAM_NAME, "memory_transfer_node::memset");
  PrepareNotify.addMetadata([&](auto TEvent) {
    xpti::addMetadata(TEvent, "sycl_device",
                      reinterpret_cast<size_t>(MDevice->getHandleRef()));
    xpti::addMetadata(TEvent, "memory_ptr", reinterpret_cast<size_t>(Ptr));
    xpti::addMetadata(TEvent, "value_set", Value);
    xpti::addMetadata(TEvent, "memory_size", Count);
    xpti::addMetadata(TEvent, "queue_id", MQueueID);
  });
  // Before we notifiy the subscribers, we broadcast the 'queue_id', which was a
  // metadata entry to TLS for use by callback handlers
  xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, MQueueID);
  // Notify XPTI about the memset submission
  PrepareNotify.notify();
  // Emit a begin/end scope for this call
  PrepareNotify.scopedNotify((uint16_t)xpti::trace_point_type_t::task_begin);
#endif
  const std::vector<unsigned char> Pattern{static_cast<unsigned char>(Value)};
  return submitMemOpHelper(
      Self, DepEvents, CallerNeedsEvent,
      [&](handler &CGH) { CGH.memset(Ptr, Value, Count); },
      [](const auto &...Args) { MemoryManager::fill_usm(Args...); }, Ptr, Self,
      Count, Pattern);
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

event queue_impl::memcpy(const std::shared_ptr<detail::queue_impl> &Self,
                         void *Dest, const void *Src, size_t Count,
                         const std::vector<event> &DepEvents,
                         bool CallerNeedsEvent, const code_location &CodeLoc) {
#if XPTI_ENABLE_INSTRUMENTATION
  // We need a code pointer value and we duse the object ptr; If code location
  // is available, we use the source file information along with the object
  // pointer.
  XPTIScope PrepareNotify((void *)this,
                          (uint16_t)xpti::trace_point_type_t::node_create,
                          SYCL_STREAM_NAME, "memory_transfer_node::memcpy");
  PrepareNotify.addMetadata([&](auto TEvent) {
    xpti::addMetadata(TEvent, "sycl_device",
                      reinterpret_cast<size_t>(MDevice->getHandleRef()));
    xpti::addMetadata(TEvent, "src_memory_ptr", reinterpret_cast<size_t>(Src));
    xpti::addMetadata(TEvent, "dest_memory_ptr",
                      reinterpret_cast<size_t>(Dest));
    xpti::addMetadata(TEvent, "memory_size", Count);
    xpti::addMetadata(TEvent, "queue_id", MQueueID);
  });
  xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, MQueueID);
  // Notify XPTI about the memcpy submission
  PrepareNotify.notify();
  // Emit a begin/end scope for this call
  PrepareNotify.scopedNotify((uint16_t)xpti::trace_point_type_t::task_begin);
#endif

  if ((!Src || !Dest) && Count != 0) {
    report(CodeLoc);
    throw exception(make_error_code(errc::invalid),
                    "NULL pointer argument in memory copy operation.");
  }
  return submitMemOpHelper(
      Self, DepEvents, CallerNeedsEvent,
      [&](handler &CGH) { CGH.memcpy(Dest, Src, Count); },
      [](const auto &...Args) { MemoryManager::copy_usm(Args...); }, Src, Self,
      Count, Dest);
}

event queue_impl::mem_advise(const std::shared_ptr<detail::queue_impl> &Self,
                             const void *Ptr, size_t Length,
                             ur_usm_advice_flags_t Advice,
                             const std::vector<event> &DepEvents,
                             bool CallerNeedsEvent) {
  return submitMemOpHelper(
      Self, DepEvents, CallerNeedsEvent,
      [&](handler &CGH) { CGH.mem_advise(Ptr, Length, Advice); },
      [](const auto &...Args) { MemoryManager::advise_usm(Args...); }, Ptr,
      Self, Length, Advice);
}

event queue_impl::memcpyToDeviceGlobal(
    const std::shared_ptr<detail::queue_impl> &Self, void *DeviceGlobalPtr,
    const void *Src, bool IsDeviceImageScope, size_t NumBytes, size_t Offset,
    const std::vector<event> &DepEvents, bool CallerNeedsEvent) {
  return submitMemOpHelper(
      Self, DepEvents, CallerNeedsEvent,
      [&](handler &CGH) {
        CGH.memcpyToDeviceGlobal(DeviceGlobalPtr, Src, IsDeviceImageScope,
                                 NumBytes, Offset);
      },
      [](const auto &...Args) {
        MemoryManager::copy_to_device_global(Args...);
      },
      DeviceGlobalPtr, IsDeviceImageScope, Self, NumBytes, Offset, Src);
}

event queue_impl::memcpyFromDeviceGlobal(
    const std::shared_ptr<detail::queue_impl> &Self, void *Dest,
    const void *DeviceGlobalPtr, bool IsDeviceImageScope, size_t NumBytes,
    size_t Offset, const std::vector<event> &DepEvents, bool CallerNeedsEvent) {
  return submitMemOpHelper(
      Self, DepEvents, CallerNeedsEvent,
      [&](handler &CGH) {
        CGH.memcpyFromDeviceGlobal(Dest, DeviceGlobalPtr, IsDeviceImageScope,
                                   NumBytes, Offset);
      },
      [](const auto &...Args) {
        MemoryManager::copy_from_device_global(Args...);
      },
      DeviceGlobalPtr, IsDeviceImageScope, Self, NumBytes, Offset, Dest);
}

event queue_impl::getLastEvent() {
  {
    // The external event is required to finish last if set, so it is considered
    // the last event if present.
    std::lock_guard<std::mutex> Lock(MInOrderExternalEventMtx);
    if (MInOrderExternalEvent)
      return *MInOrderExternalEvent;
  }

  std::lock_guard<std::mutex> Lock{MMutex};
  if (MDiscardEvents)
    return createDiscardedEvent();
  if (!MGraph.expired() && MExtGraphDeps.LastEventPtr)
    return detail::createSyclObjFromImpl<event>(MExtGraphDeps.LastEventPtr);
  if (!MDefaultGraphDeps.LastEventPtr)
    MDefaultGraphDeps.LastEventPtr = std::make_shared<event_impl>(std::nullopt);
  return detail::createSyclObjFromImpl<event>(MDefaultGraphDeps.LastEventPtr);
}

void queue_impl::addEvent(const event &Event) {
  EventImplPtr EImpl = getSyclObjImpl(Event);
  assert(EImpl && "Event implementation is missing");
  auto *Cmd = static_cast<Command *>(EImpl->getCommand());
  if (!Cmd) {
    // if there is no command on the event, we cannot track it with MEventsWeak
    // as that will leave it with no owner. Track in MEventsShared only if we're
    // unable to call urQueueFinish during wait.
    if (MEmulateOOO)
      addSharedEvent(Event);
  }
  // As long as the queue supports urQueueFinish we only need to store events
  // for unenqueued commands and host tasks.
  else if (MEmulateOOO || EImpl->getHandle() == nullptr) {
    std::weak_ptr<event_impl> EventWeakPtr{EImpl};
    std::lock_guard<std::mutex> Lock{MMutex};
    MEventsWeak.push_back(std::move(EventWeakPtr));
  }
}

/// addSharedEvent - queue_impl tracks events with weak pointers
/// but some events have no other owner. In this case,
/// addSharedEvent will have the queue track the events via a shared pointer.
void queue_impl::addSharedEvent(const event &Event) {
  assert(MEmulateOOO);
  std::lock_guard<std::mutex> Lock(MMutex);
  // Events stored in MEventsShared are not released anywhere else aside from
  // calls to queue::wait/wait_and_throw, which a user application might not
  // make, and ~queue_impl(). If the number of events grows large enough,
  // there's a good chance that most of them are already completed and ownership
  // of them can be released.
  const size_t EventThreshold = 128;
  if (MEventsShared.size() >= EventThreshold) {
    // Generally, the vector is ordered so that the oldest events are in the
    // front and the newer events are in the end.  So, search to find the first
    // event that isn't yet complete.  All the events prior to that can be
    // erased. This could leave some few events further on that have completed
    // not yet erased, but that is OK.  This cleanup doesn't have to be perfect.
    // This also keeps the algorithm linear rather than quadratic because it
    // doesn't continually recheck things towards the back of the list that
    // really haven't had time to complete.
    MEventsShared.erase(
        MEventsShared.begin(),
        std::find_if(
            MEventsShared.begin(), MEventsShared.end(), [](const event &E) {
              return E.get_info<info::event::command_execution_status>() !=
                     info::event_command_status::complete;
            }));
  }
  MEventsShared.push_back(Event);
}

event queue_impl::submit_impl(const std::function<void(handler &)> &CGF,
                              const std::shared_ptr<queue_impl> &Self,
                              const std::shared_ptr<queue_impl> &PrimaryQueue,
                              const std::shared_ptr<queue_impl> &SecondaryQueue,
                              bool CallerNeedsEvent,
                              const detail::code_location &Loc,
                              const SubmitPostProcessF *PostProcess) {
  handler Handler(Self, PrimaryQueue, SecondaryQueue, CallerNeedsEvent);
  Handler.saveCodeLoc(Loc);

  {
    NestedCallsTracker tracker;
    CGF(Handler);
  }

  // Scheduler will later omit events, that are not required to execute tasks.
  // Host and interop tasks, however, are not submitted to low-level runtimes
  // and require separate dependency management.
  const CGType Type = detail::getSyclObjImpl(Handler)->MCGType;
  event Event = detail::createSyclObjFromImpl<event>(
      std::make_shared<detail::event_impl>());
  std::vector<StreamImplPtr> Streams;
  if (Type == CGType::Kernel)
    Streams = std::move(Handler.MStreamStorage);

  if (PostProcess) {
    bool IsKernel = Type == CGType::Kernel;
    bool KernelUsesAssert = false;

    if (IsKernel)
      // Kernel only uses assert if it's non interop one
      KernelUsesAssert = !(Handler.MKernel && Handler.MKernel->isInterop()) &&
                         ProgramManager::getInstance().kernelUsesAssert(
                             Handler.MKernelName.c_str());
    finalizeHandler(Handler, Event);

    (*PostProcess)(IsKernel, KernelUsesAssert, Event);
  } else
    finalizeHandler(Handler, Event);

  addEvent(Event);

  auto EventImpl = detail::getSyclObjImpl(Event);
  for (auto &Stream : Streams) {
    // We don't want stream flushing to be blocking operation that is why submit
    // a host task to print stream buffer. It will fire up as soon as the kernel
    // finishes execution.
    event FlushEvent = submit_impl(
        [&](handler &ServiceCGH) { Stream->generateFlushCommand(ServiceCGH); },
        Self, PrimaryQueue, SecondaryQueue, /*CallerNeedsEvent*/ true, Loc, {});
    EventImpl->attachEventToCompleteWeak(detail::getSyclObjImpl(FlushEvent));
    registerStreamServiceEvent(detail::getSyclObjImpl(FlushEvent));
  }

  return Event;
}

template <typename HandlerFuncT>
event queue_impl::submitWithHandler(const std::shared_ptr<queue_impl> &Self,
                                    const std::vector<event> &DepEvents,
                                    HandlerFuncT HandlerFunc) {
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvents);
        HandlerFunc(CGH);
      },
      Self, {});
}

template <typename HandlerFuncT, typename MemOpFuncT, typename... MemOpArgTs>
event queue_impl::submitMemOpHelper(const std::shared_ptr<queue_impl> &Self,
                                    const std::vector<event> &DepEvents,
                                    bool CallerNeedsEvent,
                                    HandlerFuncT HandlerFunc,
                                    MemOpFuncT MemOpFunc,
                                    MemOpArgTs... MemOpArgs) {
  // We need to submit command and update the last event under same lock if we
  // have in-order queue.
  {
    std::unique_lock<std::mutex> Lock(MMutex, std::defer_lock);

    std::vector<event> MutableDepEvents;
    const std::vector<event> &ExpandedDepEvents =
        getExtendDependencyList(DepEvents, MutableDepEvents, Lock);

    // If we have a command graph set we need to capture the op through the
    // handler rather than by-passing the scheduler.
    if (MGraph.expired() && Scheduler::areEventsSafeForSchedulerBypass(
                                ExpandedDepEvents, MContext)) {
      if ((MDiscardEvents || !CallerNeedsEvent) &&
          supportsDiscardingPiEvents()) {
        NestedCallsTracker tracker;
        MemOpFunc(MemOpArgs..., getUrEvents(ExpandedDepEvents),
                  /*PiEvent*/ nullptr, /*EventImplPtr*/ nullptr);
        return createDiscardedEvent();
      }

      event ResEvent = prepareSYCLEventAssociatedWithQueue(Self);
      auto EventImpl = detail::getSyclObjImpl(ResEvent);
      {
        NestedCallsTracker tracker;
        ur_event_handle_t UREvent = nullptr;
        MemOpFunc(MemOpArgs..., getUrEvents(ExpandedDepEvents), &UREvent,
                  EventImpl);
        EventImpl->setHandle(UREvent);
      }

      if (isInOrder()) {
        auto &EventToStoreIn = MGraph.expired() ? MDefaultGraphDeps.LastEventPtr
                                                : MExtGraphDeps.LastEventPtr;
        EventToStoreIn = EventImpl;
      }
      // Track only if we won't be able to handle it with urQueueFinish.
      if (MEmulateOOO)
        addSharedEvent(ResEvent);
      return discard_or_return(ResEvent);
    }
  }
  return submitWithHandler(Self, DepEvents, HandlerFunc);
}

void *queue_impl::instrumentationProlog(const detail::code_location &CodeLoc,
                                        std::string &Name, int32_t StreamID,
                                        uint64_t &IId) {
  void *TraceEvent = nullptr;
  (void)CodeLoc;
  (void)Name;
  (void)StreamID;
  (void)IId;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType = xpti::trace_wait_begin;
  if (!xptiCheckTraceEnabled(StreamID, NotificationTraceType))
    return TraceEvent;

  xpti::payload_t Payload;
  bool HasSourceInfo = false;
  // We try to create a unique string for the wait() call by combining it with
  // the queue address
  xpti::utils::StringHelper NG;
  Name = NG.nameWithAddress<queue_impl *>("queue.wait", this);

  if (CodeLoc.fileName()) {
    // We have source code location information
    Payload =
        xpti::payload_t(Name.c_str(), CodeLoc.fileName(), CodeLoc.lineNumber(),
                        CodeLoc.columnNumber(), (void *)this);
    HasSourceInfo = true;
  } else {
    // We have no location information, so we'll use the address of the queue
    Payload = xpti::payload_t(Name.c_str(), (void *)this);
  }
  // wait() calls could be at different user-code locations; We create a new
  // event based on the code location info and if this has been seen before, a
  // previously created event will be returned.
  uint64_t QWaitInstanceNo = 0;
  xpti::trace_event_data_t *WaitEvent =
      xptiMakeEvent(Name.c_str(), &Payload, xpti::trace_graph_event,
                    xpti_at::active, &QWaitInstanceNo);
  IId = QWaitInstanceNo;
  if (WaitEvent) {
    xpti::addMetadata(WaitEvent, "sycl_device_type", queueDeviceToString(this));
    if (HasSourceInfo) {
      xpti::addMetadata(WaitEvent, "sym_function_name", CodeLoc.functionName());
      xpti::addMetadata(WaitEvent, "sym_source_file_name", CodeLoc.fileName());
      xpti::addMetadata(WaitEvent, "sym_line_no",
                        static_cast<int32_t>((CodeLoc.lineNumber())));
      xpti::addMetadata(WaitEvent, "sym_column_no",
                        static_cast<int32_t>((CodeLoc.columnNumber())));
    }
    xptiNotifySubscribers(StreamID, xpti::trace_wait_begin, nullptr, WaitEvent,
                          QWaitInstanceNo,
                          static_cast<const void *>(Name.c_str()));
    TraceEvent = (void *)WaitEvent;
  }
#endif
  return TraceEvent;
}

void queue_impl::instrumentationEpilog(void *TelemetryEvent, std::string &Name,
                                       int32_t StreamID, uint64_t IId) {
  (void)TelemetryEvent;
  (void)Name;
  (void)StreamID;
  (void)IId;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType = xpti::trace_wait_end;
  if (!(xptiCheckTraceEnabled(StreamID, NotificationTraceType) &&
        TelemetryEvent))
    return;
  // Close the wait() scope
  xpti::trace_event_data_t *TraceEvent =
      (xpti::trace_event_data_t *)TelemetryEvent;
  xptiNotifySubscribers(StreamID, NotificationTraceType, nullptr, TraceEvent,
                        IId, static_cast<const void *>(Name.c_str()));
#endif
}

void queue_impl::wait(const detail::code_location &CodeLoc) {
  (void)CodeLoc;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  void *TelemetryEvent = nullptr;
  uint64_t IId;
  std::string Name;
  int32_t StreamID = xptiRegisterStream(SYCL_STREAM_NAME);
  TelemetryEvent = instrumentationProlog(CodeLoc, Name, StreamID, IId);
#endif

  if (MGraph.lock()) {
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
    if (!MEventsWeak.empty() || !MEventsShared.empty()) {
      std::lock_guard<std::mutex> Lock(MMutex);
      MEventsWeak.clear();
      MEventsShared.clear();
    }
    if (!MStreamsServiceEvents.empty()) {
      std::lock_guard<std::mutex> Lock(MStreamsServiceEventsMutex);
      MStreamsServiceEvents.clear();
    }
  }

  std::vector<std::weak_ptr<event_impl>> WeakEvents;
  std::vector<event> SharedEvents;
  {
    std::lock_guard<std::mutex> Lock(MMutex);
    WeakEvents.swap(MEventsWeak);
    SharedEvents.swap(MEventsShared);

    {
      std::lock_guard<std::mutex> RequestLock(MMissedCleanupRequestsMtx);
      for (auto &UpdatedGraph : MMissedCleanupRequests)
        doUnenqueuedCommandCleanup(UpdatedGraph);
      MMissedCleanupRequests.clear();
    }
  }
  // If the queue is either a host one or does not support OOO (and we use
  // multiple in-order queues as a result of that), wait for each event
  // directly. Otherwise, only wait for unenqueued or host task events, starting
  // from the latest submitted task in order to minimize total amount of calls,
  // then handle the rest with urQueueFinish.
  const bool SupportsPiFinish = !MEmulateOOO;
  for (auto EventImplWeakPtrIt = WeakEvents.rbegin();
       EventImplWeakPtrIt != WeakEvents.rend(); ++EventImplWeakPtrIt) {
    if (std::shared_ptr<event_impl> EventImplSharedPtr =
            EventImplWeakPtrIt->lock()) {
      // A nullptr UR event indicates that urQueueFinish will not cover it,
      // either because it's a host task event or an unenqueued one.
      if (!SupportsPiFinish || nullptr == EventImplSharedPtr->getHandle()) {
        EventImplSharedPtr->wait(EventImplSharedPtr);
      }
    }
  }
  if (SupportsPiFinish) {
    const PluginPtr &Plugin = getPlugin();
    Plugin->call<UrApiKind::urQueueFinish>(getHandleRef());
    assert(SharedEvents.empty() && "Queues that support calling piQueueFinish "
                                   "shouldn't have shared events");
  } else {
    for (event &Event : SharedEvents)
      Event.wait();
  }

  std::vector<EventImplPtr> StreamsServiceEvents;
  {
    std::lock_guard<std::mutex> Lock(MStreamsServiceEventsMutex);
    StreamsServiceEvents.swap(MStreamsServiceEvents);
  }
  for (const EventImplPtr &Event : StreamsServiceEvents)
    Event->wait(Event);

#ifdef XPTI_ENABLE_INSTRUMENTATION
  instrumentationEpilog(TelemetryEvent, Name, StreamID, IId);
#endif
}

void queue_impl::constructorNotification() {
#if XPTI_ENABLE_INSTRUMENTATION
  if (xptiTraceEnabled()) {
    MStreamID = xptiRegisterStream(SYCL_STREAM_NAME);
    constexpr uint16_t NotificationTraceType =
        static_cast<uint16_t>(xpti::trace_point_type_t::queue_create);
    if (xptiCheckTraceEnabled(MStreamID, NotificationTraceType)) {
      xpti::utils::StringHelper SH;
      std::string AddrStr = SH.addressAsString<size_t>(MQueueID);
      std::string QueueName = SH.nameWithAddressString("queue", AddrStr);
      // Create a payload for the queue create event as we do not get code
      // location for the queue create event
      xpti::payload_t QPayload(QueueName.c_str());
      MInstanceID = xptiGetUniqueId();
      uint64_t RetInstanceNo;
      xpti_td *TEvent =
          xptiMakeEvent("queue_create", &QPayload,
                        (uint16_t)xpti::trace_event_type_t::algorithm,
                        xpti_at::active, &RetInstanceNo);
      // Cache the trace event, stream id and instance IDs for the destructor
      MTraceEvent = (void *)TEvent;

      xpti::addMetadata(TEvent, "sycl_context",
                        reinterpret_cast<size_t>(MContext->getHandleRef()));
      if (MDevice) {
        xpti::addMetadata(TEvent, "sycl_device_name", MDevice->getDeviceName());
        xpti::addMetadata(TEvent, "sycl_device",
                          reinterpret_cast<size_t>(MDevice->getHandleRef()));
      }
      xpti::addMetadata(TEvent, "is_inorder", MIsInorder);
      xpti::addMetadata(TEvent, "queue_id", MQueueID);
      xpti::addMetadata(TEvent, "queue_handle",
                        reinterpret_cast<size_t>(getHandleRef()));
      // Also publish to TLS before notification
      xpti::framework::stash_tuple(XPTI_QUEUE_INSTANCE_ID_KEY, MQueueID);
      xptiNotifySubscribers(
          MStreamID, (uint16_t)xpti::trace_point_type_t::queue_create, nullptr,
          TEvent, MInstanceID, static_cast<const void *>("queue_create"));
    }
  }
#endif
}

void queue_impl::destructorNotification() {
#if XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      static_cast<uint16_t>(xpti::trace_point_type_t::queue_destroy);
  if (xptiCheckTraceEnabled(MStreamID, NotificationTraceType)) {
    // Use the cached trace event, stream id and instance IDs for the
    // destructor
    xptiNotifySubscribers(MStreamID, NotificationTraceType, nullptr,
                          (xpti::trace_event_data_t *)MTraceEvent, MInstanceID,
                          static_cast<const void *>("queue_destroy"));
    xptiReleaseEvent((xpti::trace_event_data_t *)MTraceEvent);
  }
#endif
}

ur_native_handle_t queue_impl::getNative(int32_t &NativeHandleDesc) const {
  const PluginPtr &Plugin = getPlugin();
  if (getContextImplPtr()->getBackend() == backend::opencl)
    Plugin->call<UrApiKind::urQueueRetain>(MQueues[0]);
  ur_native_handle_t Handle{};
  ur_queue_native_desc_t UrNativeDesc{UR_STRUCTURE_TYPE_QUEUE_NATIVE_DESC,
                                      nullptr, nullptr};
  UrNativeDesc.pNativeData = &NativeHandleDesc;

  Plugin->call<UrApiKind::urQueueGetNativeHandle>(MQueues[0], &UrNativeDesc,
                                                  &Handle);
  return Handle;
}

bool queue_impl::ext_oneapi_empty() const {
  // If we have in-order queue where events are not discarded then just check
  // the status of the last event.
  if (isInOrder() && !MDiscardEvents) {
    std::lock_guard<std::mutex> Lock(MMutex);
    // If there is no last event we know that no work has been submitted, so it
    // must be trivially empty.
    if (!MDefaultGraphDeps.LastEventPtr)
      return true;
    // Otherwise, check if the last event is finished.
    // Note that we fall back to the backend query if the event was discarded,
    // which may happend despite the queue not being a discard event queue.
    if (!MDefaultGraphDeps.LastEventPtr->isDiscarded())
      return MDefaultGraphDeps.LastEventPtr
                 ->get_info<info::event::command_execution_status>() ==
             info::event_command_status::complete;
  }

  // Check the status of the backend queue if this is not a host queue.
  ur_bool_t IsReady = false;
  getPlugin()->call<UrApiKind::urQueueGetInfo>(
      MQueues[0], UR_QUEUE_INFO_EMPTY, sizeof(IsReady), &IsReady, nullptr);
  if (!IsReady)
    return false;

  // We may have events like host tasks which are not submitted to the backend
  // queue so we need to get their status separately.
  std::lock_guard<std::mutex> Lock(MMutex);
  for (event Event : MEventsShared)
    if (Event.get_info<info::event::command_execution_status>() !=
        info::event_command_status::complete)
      return false;

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

event queue_impl::discard_or_return(const event &Event) {
  if (!(MDiscardEvents))
    return Event;
  return createDiscardedEvent();
}

void queue_impl::revisitUnenqueuedCommandsState(
    const EventImplPtr &CompletedHostTask) {
  if (MIsInorder)
    return;
  std::unique_lock<std::mutex> Lock{MMutex, std::try_to_lock};
  if (Lock.owns_lock())
    doUnenqueuedCommandCleanup(CompletedHostTask->getCommandGraph());
  else {
    std::lock_guard<std::mutex> RequestLock(MMissedCleanupRequestsMtx);
    MMissedCleanupRequests.push_back(CompletedHostTask->getCommandGraph());
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

} // namespace detail
} // namespace _V1
} // namespace sycl
