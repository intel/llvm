//==---------------- event_impl.cpp - SYCL event ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <detail/event_impl.hpp>
#include <detail/event_info.hpp>
#include <detail/plugin.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

#include "detail/config.hpp"

#include <chrono>

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include "xpti/xpti_trace_framework.hpp"
#include <atomic>
#include <detail/xpti_registry.hpp>
#include <sstream>
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
extern xpti::trace_event_data_t *GSYCLGraphEvent;
#endif

// Threat all devices that don't support interoperability as host devices to
// avoid attempts to call method get on such events.
bool event_impl::is_host() const { return MHostEvent || !MOpenCLInterop; }

cl_event event_impl::get() const {
  if (!MOpenCLInterop) {
    throw invalid_object_error(
        "This instance of event doesn't support OpenCL interoperability.",
        PI_INVALID_EVENT);
  }
  getPlugin().call<PiApiKind::piEventRetain>(MEvent);
  return pi::cast<cl_event>(MEvent);
}

event_impl::~event_impl() {
  if (MEvent)
    getPlugin().call<PiApiKind::piEventRelease>(MEvent);
}

void event_impl::waitInternal() const {
  if (!MHostEvent && MEvent) {
    getPlugin().call<PiApiKind::piEventsWait>(1, &MEvent);
    return;
  }

  if (MState == HES_Discarded)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "waitInternal method cannot be used for a discarded event.");

  while (MState != HES_Complete)
    ;
}

void event_impl::setComplete() {
  if (MHostEvent || !MEvent) {
#ifndef NDEBUG
    int Expected = HES_NotComplete;
    int Desired = HES_Complete;

    bool Succeeded = MState.compare_exchange_strong(Expected, Desired);

    assert(Succeeded && "Unexpected state of event");
#else
    MState.store(static_cast<int>(HES_Complete));
#endif
    return;
  }

  assert(false && "setComplete is not supported for non-host event");
}

const RT::PiEvent &event_impl::getHandleRef() const { return MEvent; }
RT::PiEvent &event_impl::getHandleRef() { return MEvent; }

const ContextImplPtr &event_impl::getContextImpl() { return MContext; }

const plugin &event_impl::getPlugin() const { return MContext->getPlugin(); }

void event_impl::setContextImpl(const ContextImplPtr &Context) {
  MHostEvent = Context->is_host();
  MOpenCLInterop = !MHostEvent;
  MContext = Context;

  MState = HES_NotComplete;
}

event_impl::event_impl(HostEventState State)
    :isInited(false), MIsFlushed(true), MState(State) {}

event_impl::event_impl(RT::PiEvent Event, const context &SyclContext)
    : MEvent(Event), MContext(detail::getSyclObjImpl(SyclContext)),
      MOpenCLInterop(true), MHostEvent(false), MIsFlushed(true),
      MState(HES_Complete) {

  if (MContext->is_host()) {
    throw cl::sycl::invalid_parameter_error(
        "The syclContext must match the OpenCL context associated with the "
        "clEvent.",
        PI_INVALID_CONTEXT);
  }

  RT::PiContext TempContext;
  getPlugin().call<PiApiKind::piEventGetInfo>(MEvent, PI_EVENT_INFO_CONTEXT,
                                              sizeof(RT::PiContext),
                                              &TempContext, nullptr);
  if (MContext->getHandleRef() != TempContext) {
    throw cl::sycl::invalid_parameter_error(
        "The syclContext must match the OpenCL context associated with the "
        "clEvent.",
        PI_INVALID_CONTEXT);
  }

  getPlugin().call<PiApiKind::piEventRetain>(MEvent);
}

event_impl::event_impl(const QueueImplPtr &Queue) : MQueue{Queue} {
  if (Queue->is_host()) {
    MState.store(HES_NotComplete);

    if (Queue->has_property<property::queue::enable_profiling>()) {
      MHostProfilingInfo.reset(new HostProfilingInfo());
      if (!MHostProfilingInfo)
        throw runtime_error("Out of host memory", PI_OUT_OF_HOST_MEMORY);
    }

    return;
  }

  MState.store(HES_Complete);
}

void *event_impl::instrumentationProlog(std::string &Name, int32_t StreamID,
                                        uint64_t &IId) const {
  void *TraceEvent = nullptr;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return TraceEvent;
  // Use a thread-safe counter to get a unique instance ID for the wait() on the
  // event
  static std::atomic<uint64_t> InstanceID = {1};
  xpti::trace_event_data_t *WaitEvent = nullptr;

  // Create a string with the event address so it
  // can be associated with other debug data
  xpti::utils::StringHelper SH;
  Name = SH.nameWithAddress<RT::PiEvent>("event.wait", MEvent);

  // We can emit the wait associated with the graph if the
  // event does not have a command object or associated with
  // the command object, if it exists
  if (MCommand) {
    Command *Cmd = (Command *)MCommand;
    WaitEvent = Cmd->MTraceEvent ? static_cast<xpti_td *>(Cmd->MTraceEvent)
                                 : GSYCLGraphEvent;
  } else
    WaitEvent = GSYCLGraphEvent;

  // Record the current instance ID for use by Epilog
  IId = InstanceID++;
  xptiNotifySubscribers(StreamID, xpti::trace_wait_begin, nullptr, WaitEvent,
                        IId, static_cast<const void *>(Name.c_str()));
  TraceEvent = (void *)WaitEvent;
#endif
  return TraceEvent;
}

void event_impl::instrumentationEpilog(void *TelemetryEvent,
                                       const std::string &Name,
                                       int32_t StreamID, uint64_t IId) const {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!(xptiTraceEnabled() && TelemetryEvent))
    return;
  // Close the wait() scope
  xpti::trace_event_data_t *TraceEvent =
      (xpti::trace_event_data_t *)TelemetryEvent;
  xptiNotifySubscribers(StreamID, xpti::trace_wait_end, nullptr, TraceEvent,
                        IId, static_cast<const void *>(Name.c_str()));
#endif
}

void event_impl::wait(
    std::shared_ptr<cl::sycl::detail::event_impl> Self) const {
  if (MState == HES_Discarded)
    throw sycl::exception(make_error_code(errc::invalid),
                          "wait method cannot be used for a discarded event.");

#ifdef XPTI_ENABLE_INSTRUMENTATION
  void *TelemetryEvent = nullptr;
  uint64_t IId;
  std::string Name;
  int32_t StreamID = xptiRegisterStream(SYCL_STREAM_NAME);
  TelemetryEvent = instrumentationProlog(Name, StreamID, IId);
#endif

  if (MEvent)
    // presence of MEvent means the command has been enqueued, so no need to
    // go via the slow path event waiting in the scheduler
    waitInternal();
  else if (MCommand)
    detail::Scheduler::getInstance().waitForEvent(Self);
  cleanupCommand(std::move(Self));

#ifdef XPTI_ENABLE_INSTRUMENTATION
  instrumentationEpilog(TelemetryEvent, Name, StreamID, IId);
#endif
}

void event_impl::wait_and_throw(
    std::shared_ptr<cl::sycl::detail::event_impl> Self) {
  Scheduler &Sched = Scheduler::getInstance();

  QueueImplPtr submittedQueue = nullptr;
  {
    Scheduler::ReadLockT Lock(Sched.MGraphLock);
    Command *Cmd = static_cast<Command *>(Self->getCommand());
    if (Cmd)
      submittedQueue = Cmd->getSubmittedQueue();
  }
  wait(Self);

  {
    Scheduler::ReadLockT Lock(Sched.MGraphLock);
    for (auto &EventImpl : getWaitList()) {
      Command *Cmd = (Command *)EventImpl->getCommand();
      if (Cmd)
        Cmd->getSubmittedQueue()->throw_asynchronous();
    }
  }
  if (submittedQueue)
    submittedQueue->throw_asynchronous();
}

void event_impl::cleanupCommand(
    std::shared_ptr<cl::sycl::detail::event_impl> Self) const {
  if (MCommand && !SYCLConfig<SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP>::get())
    detail::Scheduler::getInstance().cleanupFinishedCommands(std::move(Self));
}

template <>
cl_ulong
event_impl::get_profiling_info<info::event_profiling::command_submit>() const {
  if (!MHostEvent) {
    if (MEvent)
      return get_event_profiling_info<
          info::event_profiling::command_submit>::get(this->getHandleRef(),
                                                      this->getPlugin());
    // TODO this should throw an exception if the queue the dummy event is
    // bound to does not support profiling info.
    return 0;
  }
  if (!MHostProfilingInfo)
    throw invalid_object_error("Profiling info is not available.",
                               PI_PROFILING_INFO_NOT_AVAILABLE);
  return MHostProfilingInfo->getStartTime();
}

template <>
cl_ulong
event_impl::get_profiling_info<info::event_profiling::command_start>() const {
  if (!MHostEvent) {
    if (MEvent)
      return get_event_profiling_info<
          info::event_profiling::command_start>::get(this->getHandleRef(),
                                                     this->getPlugin());
    // TODO this should throw an exception if the queue the dummy event is
    // bound to does not support profiling info.
    return 0;
  }
  if (!MHostProfilingInfo)
    throw invalid_object_error("Profiling info is not available.",
                               PI_PROFILING_INFO_NOT_AVAILABLE);
  return MHostProfilingInfo->getStartTime();
}

template <>
cl_ulong
event_impl::get_profiling_info<info::event_profiling::command_end>() const {
  if (!MHostEvent) {
    if (MEvent)
      return get_event_profiling_info<info::event_profiling::command_end>::get(
          this->getHandleRef(), this->getPlugin());
    // TODO this should throw an exception if the queue the dummy event is
    // bound to does not support profiling info.
    return 0;
  }
  if (!MHostProfilingInfo)
    throw invalid_object_error("Profiling info is not available.",
                               PI_PROFILING_INFO_NOT_AVAILABLE);
  return MHostProfilingInfo->getEndTime();
}

template <> cl_uint event_impl::get_info<info::event::reference_count>() const {
  if (!MHostEvent && MEvent) {
    return get_event_info<info::event::reference_count>::get(
        this->getHandleRef(), this->getPlugin());
  }
  return 0;
}

template <>
info::event_command_status
event_impl::get_info<info::event::command_execution_status>() const {
  if (MState == HES_Discarded)
    return info::event_command_status::ext_oneapi_unknown;

  if (!MHostEvent && MEvent) {
    return get_event_info<info::event::command_execution_status>::get(
        this->getHandleRef(), this->getPlugin());
  }
  return MHostEvent && MState.load() != HES_Complete
             ? sycl::info::event_command_status::submitted
             : info::event_command_status::complete;
}

static uint64_t getTimestamp() {
  auto TimeStamp = std::chrono::high_resolution_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(TimeStamp)
      .count();
}

void HostProfilingInfo::start() { StartTime = getTimestamp(); }

void HostProfilingInfo::end() { EndTime = getTimestamp(); }

pi_native_handle event_impl::getNative() const {
  if (!MContext) {
    static context SyclContext;
    MContext = getSyclObjImpl(SyclContext);
    MHostEvent = MContext->is_host();
    MOpenCLInterop = !MHostEvent;
  }
  //RT::PiContext &Context = MContext->getHandleRef();
  auto Plugin = getPlugin();
  if (!isInited) {
    isInited = true;
    auto TempContext = MContext.get()->getHandleRef();
    getPlugin().call<PiApiKind::piEventCreate>(TempContext, &MEvent);
    //Plugin.call<PiApiKind::piEventCreate>(Context, getHandleRef());
  }
  if (Plugin.getBackend() == backend::opencl)
    Plugin.call<PiApiKind::piEventRetain>(getHandleRef());
  pi_native_handle Handle;
  Plugin.call<PiApiKind::piextEventGetNativeHandle>(getHandleRef(), &Handle);
  return Handle;
}

std::vector<EventImplPtr> event_impl::getWaitList() {
  if (MState == HES_Discarded)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "get_wait_list() cannot be used for a discarded event.");

  std::lock_guard<std::mutex> Lock(MMutex);

  std::vector<EventImplPtr> Result;
  Result.reserve(MPreparedDepsEvents.size() + MPreparedHostDepsEvents.size());
  Result.insert(Result.end(), MPreparedDepsEvents.begin(),
                MPreparedDepsEvents.end());
  Result.insert(Result.end(), MPreparedHostDepsEvents.begin(),
                MPreparedHostDepsEvents.end());

  return Result;
}

void event_impl::flushIfNeeded(const QueueImplPtr &UserQueue) {
  if (MIsFlushed)
    return;

  QueueImplPtr Queue = MQueue.lock();
  // If the queue has been released, all of the commands have already been
  // implicitly flushed by piQueueRelease.
  if (!Queue) {
    MIsFlushed = true;
    return;
  }
  if (Queue == UserQueue)
    return;

  // Check if the task for this event has already been submitted.
  assert(MEvent != nullptr);
  pi_event_status Status = PI_EVENT_QUEUED;
  getPlugin().call<PiApiKind::piEventGetInfo>(
      MEvent, PI_EVENT_INFO_COMMAND_EXECUTION_STATUS, sizeof(pi_int32), &Status,
      nullptr);
  if (Status == PI_EVENT_QUEUED) {
    getPlugin().call<PiApiKind::piQueueFlush>(Queue->getHandleRef());
  }
  MIsFlushed = true;
}

void event_impl::cleanupDependencyEvents() {
  std::lock_guard<std::mutex> Lock(MMutex);
  MPreparedDepsEvents.clear();
  MPreparedHostDepsEvents.clear();
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
