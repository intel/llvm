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
#include <sycl/detail/pi.hpp>
#include <sycl/device.hpp>

#include <cstring>
#include <utility>

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include "xpti/xpti_trace_framework.hpp"
#include <detail/xpti_registry.hpp>
#include <sstream>
#endif

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
template <>
uint32_t queue_impl::get_info<info::queue::reference_count>() const {
  sycl::detail::pi::PiResult result = PI_SUCCESS;
  if (!is_host())
    getPlugin()->call<PiApiKind::piQueueGetInfo>(
        MQueues[0], PI_QUEUE_INFO_REFERENCE_COUNT, sizeof(result), &result,
        nullptr);
  return result;
}

template <> context queue_impl::get_info<info::queue::context>() const {
  return get_context();
}

template <> device queue_impl::get_info<info::queue::device>() const {
  return get_device();
}

static event
prepareUSMEvent(const std::shared_ptr<detail::queue_impl> &QueueImpl,
                sycl::detail::pi::PiEvent NativeEvent) {
  auto EventImpl = std::make_shared<detail::event_impl>(QueueImpl);
  EventImpl->getHandleRef() = NativeEvent;
  EventImpl->setContextImpl(detail::getSyclObjImpl(QueueImpl->get_context()));
  EventImpl->setStateIncomplete();
  return detail::createSyclObjFromImpl<event>(EventImpl);
}

static event createDiscardedEvent() {
  EventImplPtr EventImpl =
      std::make_shared<event_impl>(event_impl::HES_Discarded);
  return createSyclObjFromImpl<event>(EventImpl);
}

event queue_impl::memset(const std::shared_ptr<detail::queue_impl> &Self,
                         void *Ptr, int Value, size_t Count,
                         const std::vector<event> &DepEvents) {
#if XPTI_ENABLE_INSTRUMENTATION
  // We need a code pointer value and we use the object ptr; if code location
  // information is available, we will have function name and source file
  // information
  XPTIScope PrepareNotify((void *)this,
                          (uint16_t)xpti::trace_point_type_t::node_create,
                          SYCL_MEM_ALLOC_STREAM_NAME, "queue.memset()");
  PrepareNotify.addMetadata([&](auto TEvent) {
    xpti::addMetadata(TEvent, "sycl_device",
                      reinterpret_cast<size_t>(
                          MDevice->is_host() ? 0 : MDevice->getHandleRef()));
    xpti::addMetadata(TEvent, "memory_ptr", reinterpret_cast<size_t>(Ptr));
    xpti::addMetadata(TEvent, "value_set", Value);
    xpti::addMetadata(TEvent, "memory_size", Count);
  });
  // Notify XPTI about the memset submission
  PrepareNotify.notify();
  // Emit a begin/end scope for this call
  PrepareNotify.scopedNotify((uint16_t)xpti::trace_point_type_t::task_begin);
#endif
  if (MHasDiscardEventsSupport) {
    MemoryManager::fill_usm(Ptr, Self, Count, Value,
                            getOrWaitEvents(DepEvents, MContext), nullptr);
    return createDiscardedEvent();
  }
  event ResEvent;
  {
    // We need to submit command and update the last event under same lock if we
    // have in-order queue.
    auto ScopeLock = isInOrder() ? std::unique_lock<std::mutex>(MLastEventMtx)
                                 : std::unique_lock<std::mutex>();
    // If the last submitted command in the in-order queue is host_task then
    // wait for it before submitting usm command.
    if (isInOrder() && MLastCGType == CG::CGTYPE::CodeplayHostTask)
      MLastEvent.wait();

    sycl::detail::pi::PiEvent NativeEvent{};
    MemoryManager::fill_usm(Ptr, Self, Count, Value,
                            getOrWaitEvents(DepEvents, MContext), &NativeEvent);

    if (MContext->is_host())
      return MDiscardEvents ? createDiscardedEvent() : event();

    ResEvent = prepareUSMEvent(Self, NativeEvent);
    if (isInOrder()) {
      MLastEvent = ResEvent;
      // We don't create a command group for usm commands, so set it to None.
      // This variable is used to perform explicit dependency management when
      // required.
      MLastCGType = CG::CGTYPE::None;
    }
  }
  // Track only if we won't be able to handle it with piQueueFinish.
  if (MEmulateOOO)
    addSharedEvent(ResEvent);
  return MDiscardEvents ? createDiscardedEvent() : ResEvent;
}

event queue_impl::memcpy(const std::shared_ptr<detail::queue_impl> &Self,
                         void *Dest, const void *Src, size_t Count,
                         const std::vector<event> &DepEvents) {
#if XPTI_ENABLE_INSTRUMENTATION
  // We need a code pointer value and we duse the object ptr; If code location
  // is available, we use the source file information along with the object
  // pointer.
  XPTIScope PrepareNotify((void *)this,
                          (uint16_t)xpti::trace_point_type_t::node_create,
                          SYCL_MEM_ALLOC_STREAM_NAME, "queue.memcpy()");
  PrepareNotify.addMetadata([&](auto TEvent) {
    xpti::addMetadata(TEvent, "sycl_device",
                      reinterpret_cast<size_t>(
                          MDevice->is_host() ? 0 : MDevice->getHandleRef()));
    xpti::addMetadata(TEvent, "src_memory_ptr", reinterpret_cast<size_t>(Src));
    xpti::addMetadata(TEvent, "dest_memory_ptr",
                      reinterpret_cast<size_t>(Dest));
    xpti::addMetadata(TEvent, "memory_size", Count);
  });
  // Notify XPTI about the memset submission
  PrepareNotify.notify();
  // Emit a begin/end scope for this call
  PrepareNotify.scopedNotify((uint16_t)xpti::trace_point_type_t::task_begin);
#endif
  // If we have a command graph set we need to capture the copy through normal
  // queue submission rather than execute the copy directly.
  if (MGraph) {
    return submit(
        [&](handler &CGH) {
          CGH.depends_on(DepEvents);
          CGH.memcpy(Dest, Src, Count);
        },
        Self, {});
  }
  if (MHasDiscardEventsSupport) {
    MemoryManager::copy_usm(Src, Self, Count, Dest,
                            getOrWaitEvents(DepEvents, MContext), nullptr);
    return createDiscardedEvent();
  }
  event ResEvent;
  {
    // We need to submit command and update the last event under same lock if we
    // have in-order queue.
    auto ScopeLock = isInOrder() ? std::unique_lock<std::mutex>(MLastEventMtx)
                                 : std::unique_lock<std::mutex>();
    // If the last submitted command in the in-order queue is host_task then
    // wait for it before submitting usm command.
    if (isInOrder() && MLastCGType == CG::CGTYPE::CodeplayHostTask)
      MLastEvent.wait();

    sycl::detail::pi::PiEvent NativeEvent{};
    MemoryManager::copy_usm(Src, Self, Count, Dest,
                            getOrWaitEvents(DepEvents, MContext), &NativeEvent);

    if (MContext->is_host())
      return MDiscardEvents ? createDiscardedEvent() : event();

    ResEvent = prepareUSMEvent(Self, NativeEvent);
    if (isInOrder()) {
      MLastEvent = ResEvent;
      // We don't create a command group for usm commands, so set it to None.
      // This variable is used to perform explicit dependency management when
      // required.
      MLastCGType = CG::CGTYPE::None;
    }
  }
  // Track only if we won't be able to handle it with piQueueFinish.
  if (MEmulateOOO)
    addSharedEvent(ResEvent);
  return MDiscardEvents ? createDiscardedEvent() : ResEvent;
}

event queue_impl::mem_advise(const std::shared_ptr<detail::queue_impl> &Self,
                             const void *Ptr, size_t Length,
                             pi_mem_advice Advice,
                             const std::vector<event> &DepEvents) {
  if (MHasDiscardEventsSupport) {
    MemoryManager::advise_usm(Ptr, Self, Length, Advice,
                              getOrWaitEvents(DepEvents, MContext), nullptr);
    return createDiscardedEvent();
  }
  event ResEvent;
  {
    // We need to submit command and update the last event under same lock if we
    // have in-order queue.
    auto ScopeLock = isInOrder() ? std::unique_lock<std::mutex>(MLastEventMtx)
                                 : std::unique_lock<std::mutex>();
    // If the last submitted command in the in-order queue is host_task then
    // wait for it before submitting usm command.
    if (isInOrder() && MLastCGType == CG::CGTYPE::CodeplayHostTask)
      MLastEvent.wait();

    sycl::detail::pi::PiEvent NativeEvent{};
    MemoryManager::advise_usm(Ptr, Self, Length, Advice,
                              getOrWaitEvents(DepEvents, MContext),
                              &NativeEvent);

    if (MContext->is_host())
      return MDiscardEvents ? createDiscardedEvent() : event();

    ResEvent = prepareUSMEvent(Self, NativeEvent);
    if (isInOrder()) {
      MLastEvent = ResEvent;
      // We don't create a command group for usm commands, so set it to None.
      // This variable is used to perform explicit dependency management when
      // required.
      MLastCGType = CG::CGTYPE::None;
    }
  }
  // Track only if we won't be able to handle it with piQueueFinish.
  if (MEmulateOOO)
    addSharedEvent(ResEvent);
  return MDiscardEvents ? createDiscardedEvent() : ResEvent;
}

event queue_impl::memcpyToDeviceGlobal(
    const std::shared_ptr<detail::queue_impl> &Self, void *DeviceGlobalPtr,
    const void *Src, bool IsDeviceImageScope, size_t NumBytes, size_t Offset,
    const std::vector<event> &DepEvents) {
  if (MHasDiscardEventsSupport) {
    MemoryManager::copy_to_device_global(
        DeviceGlobalPtr, IsDeviceImageScope, Self, NumBytes, Offset, Src,
        getOrWaitEvents(DepEvents, MContext), nullptr);
    return createDiscardedEvent();
  }
  event ResEvent;
  {
    // We need to submit command and update the last event under same lock if we
    // have in-order queue.
    auto ScopeLock = isInOrder() ? std::unique_lock<std::mutex>(MLastEventMtx)
                                 : std::unique_lock<std::mutex>();
    // If the last submitted command in the in-order queue is host_task then
    // wait for it before submitting usm command.
    if (isInOrder() && MLastCGType == CG::CGTYPE::CodeplayHostTask)
      MLastEvent.wait();

    sycl::detail::pi::PiEvent NativeEvent{};
    MemoryManager::copy_to_device_global(
        DeviceGlobalPtr, IsDeviceImageScope, Self, NumBytes, Offset, Src,
        getOrWaitEvents(DepEvents, MContext), &NativeEvent);

    if (MContext->is_host())
      return MDiscardEvents ? createDiscardedEvent() : event();

    ResEvent = prepareUSMEvent(Self, NativeEvent);

    if (isInOrder()) {
      MLastEvent = ResEvent;
      // We don't create a command group for usm commands, so set it to None.
      // This variable is used to perform explicit dependency management when
      // required.
      MLastCGType = CG::CGTYPE::None;
    }
  }
  // Track only if we won't be able to handle it with piQueueFinish.
  if (MEmulateOOO)
    addSharedEvent(ResEvent);
  return MDiscardEvents ? createDiscardedEvent() : ResEvent;
}

event queue_impl::memcpyFromDeviceGlobal(
    const std::shared_ptr<detail::queue_impl> &Self, void *Dest,
    const void *DeviceGlobalPtr, bool IsDeviceImageScope, size_t NumBytes,
    size_t Offset, const std::vector<event> &DepEvents) {
  if (MHasDiscardEventsSupport) {
    MemoryManager::copy_from_device_global(
        DeviceGlobalPtr, IsDeviceImageScope, Self, NumBytes, Offset, Dest,
        getOrWaitEvents(DepEvents, MContext), nullptr);
    return createDiscardedEvent();
  }
  event ResEvent;
  {
    // We need to submit command and update the last event under same lock if we
    // have in-order queue.
    auto ScopeLock = isInOrder() ? std::unique_lock<std::mutex>(MLastEventMtx)
                                 : std::unique_lock<std::mutex>();
    // If the last submitted command in the in-order queue is host_task then
    // wait for it before submitting usm command.
    if (isInOrder() && MLastCGType == CG::CGTYPE::CodeplayHostTask)
      MLastEvent.wait();

    sycl::detail::pi::PiEvent NativeEvent{};
    MemoryManager::copy_from_device_global(
        DeviceGlobalPtr, IsDeviceImageScope, Self, NumBytes, Offset, Dest,
        getOrWaitEvents(DepEvents, MContext), &NativeEvent);

    if (MContext->is_host())
      return MDiscardEvents ? createDiscardedEvent() : event();

    ResEvent = prepareUSMEvent(Self, NativeEvent);

    if (isInOrder()) {
      MLastEvent = ResEvent;
      // We don't create a command group for usm commands, so set it to None.
      // This variable is used to perform explicit dependency management when
      // required.
      MLastCGType = CG::CGTYPE::None;
    }
  }
  // Track only if we won't be able to handle it with piQueueFinish.
  if (MEmulateOOO)
    addSharedEvent(ResEvent);
  return MDiscardEvents ? createDiscardedEvent() : ResEvent;
}

void queue_impl::addEvent(const event &Event) {
  EventImplPtr EImpl = getSyclObjImpl(Event);
  assert(EImpl && "Event implementation is missing");
  auto *Cmd = static_cast<Command *>(EImpl->getCommand());
  if (!Cmd) {
    // if there is no command on the event, we cannot track it with MEventsWeak
    // as that will leave it with no owner. Track in MEventsShared only if we're
    // unable to call piQueueFinish during wait.
    if (is_host() || MEmulateOOO)
      addSharedEvent(Event);
  }
  // As long as the queue supports piQueueFinish we only need to store events
  // for unenqueued commands and host tasks.
  else if (is_host() || MEmulateOOO || EImpl->getHandleRef() == nullptr) {
    std::weak_ptr<event_impl> EventWeakPtr{EImpl};
    std::lock_guard<std::mutex> Lock{MMutex};
    MEventsWeak.push_back(std::move(EventWeakPtr));
  }
}

/// addSharedEvent - queue_impl tracks events with weak pointers
/// but some events have no other owner. In this case,
/// addSharedEvent will have the queue track the events via a shared pointer.
void queue_impl::addSharedEvent(const event &Event) {
  assert(is_host() || MEmulateOOO);
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
    device D = get_device();
    std::string DevStr;
    if (getSyclObjImpl(D)->is_host())
      DevStr = "HOST";
    else if (D.is_cpu())
      DevStr = "CPU";
    else if (D.is_gpu())
      DevStr = "GPU";
    else if (D.is_accelerator())
      DevStr = "ACCELERATOR";
    else
      DevStr = "UNKNOWN";
    xpti::addMetadata(WaitEvent, "sycl_device", DevStr);
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

  if (MGraph) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "wait cannot be called for a queue which is "
                          "recording to a command graph.");
  }

  std::vector<std::weak_ptr<event_impl>> WeakEvents;
  std::vector<event> SharedEvents;
  {
    std::lock_guard<std::mutex> Lock(MMutex);
    WeakEvents.swap(MEventsWeak);
    SharedEvents.swap(MEventsShared);
  }
  // If the queue is either a host one or does not support OOO (and we use
  // multiple in-order queues as a result of that), wait for each event
  // directly. Otherwise, only wait for unenqueued or host task events, starting
  // from the latest submitted task in order to minimize total amount of calls,
  // then handle the rest with piQueueFinish.
  const bool SupportsPiFinish = !is_host() && !MEmulateOOO;
  for (auto EventImplWeakPtrIt = WeakEvents.rbegin();
       EventImplWeakPtrIt != WeakEvents.rend(); ++EventImplWeakPtrIt) {
    if (std::shared_ptr<event_impl> EventImplSharedPtr =
            EventImplWeakPtrIt->lock()) {
      // A nullptr PI event indicates that piQueueFinish will not cover it,
      // either because it's a host task event or an unenqueued one.
      if (!SupportsPiFinish || nullptr == EventImplSharedPtr->getHandleRef()) {
        EventImplSharedPtr->wait(EventImplSharedPtr);
      }
    }
  }
  if (SupportsPiFinish) {
    const PluginPtr &Plugin = getPlugin();
    Plugin->call<detail::PiApiKind::piQueueFinish>(getHandleRef());
    assert(SharedEvents.empty() && "Queues that support calling piQueueFinish "
                                   "shouldn't have shared events");
  } else {
    for (event &Event : SharedEvents)
      Event.wait();
  }

  std::vector<EventImplPtr> StreamsServiceEvents;
  {
    std::lock_guard<std::mutex> Lock(MMutex);
    StreamsServiceEvents.swap(MStreamsServiceEvents);
  }
  for (const EventImplPtr &Event : StreamsServiceEvents)
    Event->wait(Event);

#ifdef XPTI_ENABLE_INSTRUMENTATION
  instrumentationEpilog(TelemetryEvent, Name, StreamID, IId);
#endif
}

pi_native_handle queue_impl::getNative(int32_t &NativeHandleDesc) const {
  const PluginPtr &Plugin = getPlugin();
  if (getContextImplPtr()->getBackend() == backend::opencl)
    Plugin->call<PiApiKind::piQueueRetain>(MQueues[0]);
  pi_native_handle Handle{};
  Plugin->call<PiApiKind::piextQueueGetNativeHandle>(MQueues[0], &Handle,
                                                     &NativeHandleDesc);
  return Handle;
}

bool queue_impl::ext_oneapi_empty() const {
  // If we have in-order queue where events are not discarded then just check
  // the status of the last event.
  if (isInOrder() && !MDiscardEvents) {
    std::lock_guard<std::mutex> Lock(MLastEventMtx);
    return MLastEvent.get_info<info::event::command_execution_status>() ==
           info::event_command_status::complete;
  }

  // Check the status of the backend queue if this is not a host queue.
  if (!is_host()) {
    pi_bool IsReady = false;
    getPlugin()->call<PiApiKind::piQueueGetInfo>(
        MQueues[0], PI_EXT_ONEAPI_QUEUE_INFO_EMPTY, sizeof(pi_bool), &IsReady,
        nullptr);
    if (!IsReady)
      return false;
  }

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
      if (EventImplSharedPtr->is_host() &&
          EventImplSharedPtr
                  ->get_info<info::event::command_execution_status>() !=
              info::event_command_status::complete)
        return false;

  // If we didn't exit early above then it means that all events in the queue
  // are completed.
  return true;
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
