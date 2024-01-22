//==---------------- event_impl.cpp - SYCL event ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/event_impl.hpp>
#include <detail/event_info.hpp>
#include <detail/plugin.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <sycl/context.hpp>
#include <sycl/device_selector.hpp>

#include "detail/config.hpp"

#include <chrono>

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include "xpti/xpti_trace_framework.hpp"
#include <atomic>
#include <detail/xpti_registry.hpp>
#include <sstream>
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
extern xpti::trace_event_data_t *GSYCLGraphEvent;
#endif

// If we do not yet have a context, use the default one.
void event_impl::ensureContextInitialized() {
  if (MIsContextInitialized)
    return;

  if (MHostEvent) {
    QueueImplPtr HostQueue = Scheduler::getInstance().getDefaultHostQueue();
    this->setContextImpl(detail::getSyclObjImpl(HostQueue->get_context()));
  } else {
    const device SyclDevice;
    this->setContextImpl(detail::queue_impl::getDefaultOrNew(
        detail::getSyclObjImpl(SyclDevice)));
  }
}

bool event_impl::is_host() {
  // Treat all devices that don't support interoperability as host devices to
  // avoid attempts to call method get on such events.
  return MHostEvent;
}

event_impl::~event_impl() {
  if (MEvent)
    getPlugin()->call<PiApiKind::piEventRelease>(MEvent);
}

void event_impl::waitInternal() {
  if (!MHostEvent && MEvent) {
    // Wait for the native event
    getPlugin()->call<PiApiKind::piEventsWait>(1, &MEvent);
  } else if (MState == HES_Discarded) {
    // Waiting for the discarded event is invalid
    throw sycl::exception(
        make_error_code(errc::invalid),
        "waitInternal method cannot be used for a discarded event.");
  } else if (MState != HES_Complete) {
    // Wait for the host event
    std::unique_lock<std::mutex> lock(MMutex);
    cv.wait(lock, [this] { return MState == HES_Complete; });
  }

  // Wait for connected events(e.g. streams prints)
  for (const EventImplPtr &Event : MPostCompleteEvents)
    Event->wait(Event);
}

void event_impl::setComplete() {
  if (MHostEvent || !MEvent) {
    {
      std::unique_lock<std::mutex> lock(MMutex);
#ifndef NDEBUG
      int Expected = HES_NotComplete;
      int Desired = HES_Complete;

      bool Succeeded = MState.compare_exchange_strong(Expected, Desired);

      assert(Succeeded && "Unexpected state of event");
#else
      MState.store(static_cast<int>(HES_Complete));
#endif
    }
    cv.notify_all();
    return;
  }

  assert(false && "setComplete is not supported for non-host event");
}

static uint64_t inline getTimestamp() {
  auto Timestamp = std::chrono::high_resolution_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Timestamp)
      .count();
}

const sycl::detail::pi::PiEvent &event_impl::getHandleRef() const {
  return MEvent;
}
sycl::detail::pi::PiEvent &event_impl::getHandleRef() { return MEvent; }

const ContextImplPtr &event_impl::getContextImpl() {
  ensureContextInitialized();
  return MContext;
}

const PluginPtr &event_impl::getPlugin() {
  ensureContextInitialized();
  return MContext->getPlugin();
}

void event_impl::setStateIncomplete() { MState = HES_NotComplete; }

void event_impl::setContextImpl(const ContextImplPtr &Context) {
  MHostEvent = Context->is_host();
  MContext = Context;
  MIsContextInitialized = true;
}

event_impl::event_impl(sycl::detail::pi::PiEvent Event,
                       const context &SyclContext)
    : MIsContextInitialized(true), MEvent(Event),
      MContext(detail::getSyclObjImpl(SyclContext)), MHostEvent(false),
      MIsFlushed(true), MState(HES_Complete) {

  if (MContext->is_host()) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "The syclContext must match the OpenCL context "
                          "associated with the clEvent. " +
                              codeToString(PI_ERROR_INVALID_CONTEXT));
  }

  sycl::detail::pi::PiContext TempContext;
  getPlugin()->call<PiApiKind::piEventGetInfo>(
      MEvent, PI_EVENT_INFO_CONTEXT, sizeof(sycl::detail::pi::PiContext),
      &TempContext, nullptr);
  if (MContext->getHandleRef() != TempContext) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "The syclContext must match the OpenCL context "
                          "associated with the clEvent. " +
                              codeToString(PI_ERROR_INVALID_CONTEXT));
  }
}

event_impl::event_impl(const QueueImplPtr &Queue)
    : MQueue{Queue}, MIsProfilingEnabled{Queue->is_host() ||
                                         Queue->MIsProfilingEnabled},
      MFallbackProfiling{MIsProfilingEnabled && Queue->isProfilingFallback()} {
  this->setContextImpl(Queue->getContextImplPtr());
  if (Queue->is_host()) {
    MState.store(HES_NotComplete);
    if (Queue->has_property<property::queue::enable_profiling>()) {
      MHostProfilingInfo.reset(new HostProfilingInfo());
      if (!MHostProfilingInfo)
        throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                              "Out of host memory " +
                                  codeToString(PI_ERROR_OUT_OF_HOST_MEMORY));
    }
    return;
  }
  MState.store(HES_Complete);
}

void *event_impl::instrumentationProlog(std::string &Name, int32_t StreamID,
                                        uint64_t &IId) const {
  void *TraceEvent = nullptr;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType = xpti::trace_wait_begin;
  if (!xptiCheckTraceEnabled(StreamID, NotificationTraceType))
    return TraceEvent;
  // Use a thread-safe counter to get a unique instance ID for the wait() on the
  // event
  static std::atomic<uint64_t> InstanceID = {1};
  xpti::trace_event_data_t *WaitEvent = nullptr;

  // Create a string with the event address so it
  // can be associated with other debug data
  xpti::utils::StringHelper SH;
  Name = SH.nameWithAddress<sycl::detail::pi::PiEvent>("event.wait", MEvent);

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
  xptiNotifySubscribers(StreamID, NotificationTraceType, nullptr, WaitEvent,
                        IId, static_cast<const void *>(Name.c_str()));
  TraceEvent = (void *)WaitEvent;
#endif
  return TraceEvent;
}

void event_impl::instrumentationEpilog(void *TelemetryEvent,
                                       const std::string &Name,
                                       int32_t StreamID, uint64_t IId) const {
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

void event_impl::wait(std::shared_ptr<sycl::detail::event_impl> Self) {
  if (MState == HES_Discarded)
    throw sycl::exception(make_error_code(errc::invalid),
                          "wait method cannot be used for a discarded event.");

  if (MGraph.lock()) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "wait method cannot be used for an event associated "
                          "with a command graph.");
  }

#ifdef XPTI_ENABLE_INSTRUMENTATION
  void *TelemetryEvent = nullptr;
  uint64_t IId = 0;
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

#ifdef XPTI_ENABLE_INSTRUMENTATION
  instrumentationEpilog(TelemetryEvent, Name, StreamID, IId);
#endif
}

void event_impl::wait_and_throw(
    std::shared_ptr<sycl::detail::event_impl> Self) {
  wait(Self);

  if (QueueImplPtr SubmittedQueue = MSubmittedQueue.lock())
    SubmittedQueue->throw_asynchronous();
}

void event_impl::checkProfilingPreconditions() const {
  std::weak_ptr<queue_impl> EmptyPtr;

  if (!EmptyPtr.owner_before(MQueue) && !MQueue.owner_before(EmptyPtr)) {
    throw sycl::exception(make_error_code(sycl::errc::invalid),
                          "Profiling information is unavailable as the event "
                          "has no associated queue.");
  }
  if (!MIsProfilingEnabled) {
    throw sycl::exception(
        make_error_code(sycl::errc::invalid),
        "Profiling information is unavailable as the queue associated with "
        "the event does not have the 'enable_profiling' property.");
  }
}

template <>
uint64_t
event_impl::get_profiling_info<info::event_profiling::command_submit>() {
  checkProfilingPreconditions();
  // The delay between the submission and the actual start of a CommandBuffer
  // can be short. Consequently, the submission time, which is based on
  // an estimated clock and not on the real device clock, may be ahead of the
  // start time, which is based on the actual device clock.
  // MSubmitTime is set in a critical performance path.
  // Force reading the device clock when setting MSubmitTime may deteriorate
  // the performance.
  // Since submit time is an estimated time, we implement this little hack
  // that allows all profiled time to be meaningful.
  // (Note that the observed time deviation between the estimated clock and
  // the real device clock is typically less than 0.5ms. The approximation we
  // made by forcing the re-sync of submit time to start time is less than
  // 0.5ms. These timing values were obtained empirically using an integrated
  // Intel GPU).
  if (MEventFromSubmittedExecCommandBuffer && !MHostEvent && MEvent) {
    uint64_t StartTime =
        get_event_profiling_info<info::event_profiling::command_start>(
            this->getHandleRef(), this->getPlugin());
    if (StartTime < MSubmitTime)
      MSubmitTime = StartTime;
  }
  return MSubmitTime;
}

template <>
uint64_t
event_impl::get_profiling_info<info::event_profiling::command_start>() {
  checkProfilingPreconditions();
  if (!MHostEvent) {
    if (MEvent) {
      auto StartTime =
          get_event_profiling_info<info::event_profiling::command_start>(
              this->getHandleRef(), this->getPlugin());
      if (!MFallbackProfiling) {
        return StartTime;
      } else {
        auto DeviceBaseTime =
            get_event_profiling_info<info::event_profiling::command_submit>(
                this->getHandleRef(), this->getPlugin());
        return MHostBaseTime - DeviceBaseTime + StartTime;
      }
    }
    return 0;
  }
  if (!MHostProfilingInfo)
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Profiling info is not available. " +
            codeToString(PI_ERROR_PROFILING_INFO_NOT_AVAILABLE));
  return MHostProfilingInfo->getStartTime();
}

template <>
uint64_t event_impl::get_profiling_info<info::event_profiling::command_end>() {
  checkProfilingPreconditions();
  if (!MHostEvent) {
    if (MEvent) {
      auto EndTime =
          get_event_profiling_info<info::event_profiling::command_end>(
              this->getHandleRef(), this->getPlugin());
      if (!MFallbackProfiling) {
        return EndTime;
      } else {
        auto DeviceBaseTime =
            get_event_profiling_info<info::event_profiling::command_submit>(
                this->getHandleRef(), this->getPlugin());
        return MHostBaseTime - DeviceBaseTime + EndTime;
      }
    }
    return 0;
  }
  if (!MHostProfilingInfo)
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Profiling info is not available. " +
            codeToString(PI_ERROR_PROFILING_INFO_NOT_AVAILABLE));
  return MHostProfilingInfo->getEndTime();
}

template <>
uint64_t event_impl::get_profiling_info<info::event_profiling::command_submit>(
    std::shared_ptr<ext::oneapi::experimental::detail::node_impl> NodeImpl) {
  if (isEventFromSubmitedExecCommandBuffer()) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Node profiling information is only available for "
                          "event returned from graph submission" +
                              codeToString(PI_ERROR_INVALID_VALUE));
  }
  checkProfilingPreconditions();
  return MSubmitTime;
}

template <>
uint64_t event_impl::get_profiling_info<info::event_profiling::command_start>(
    std::shared_ptr<ext::oneapi::experimental::detail::node_impl> NodeImpl) {
  if (isEventFromSubmitedExecCommandBuffer()) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Node profiling information is only available for "
                          "event returned from graph submission" +
                              codeToString(PI_ERROR_INVALID_VALUE));
  }
  checkProfilingPreconditions();
  // Node profiling is only available for Level-Zero backend
  if (MContext->getBackend() != backend::ext_oneapi_level_zero) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Profiling info is not available. " +
            codeToString(PI_ERROR_PROFILING_INFO_NOT_AVAILABLE));
  }
  if (!MHostEvent) {
    if (MEvent) {
      sycl::detail::pi::PiExtSyncPoint SyncPoint =
          MExecGraph->getSyncPointFromNode(NodeImpl);

      auto StartTime =
          get_event_profiling_info<info::event_profiling::command_start>(
              this->getHandleRef(), SyncPoint, this->getPlugin());
      if (!MFallbackProfiling) {
        return StartTime;
      } else {
        auto DeviceBaseTime =
            get_event_profiling_info<info::event_profiling::command_submit>(
                this->getHandleRef(), this->getPlugin());
        return MHostBaseTime - DeviceBaseTime + StartTime;
      }
    }
    return 0;
  }
  if (!MHostProfilingInfo)
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Profiling info is not available. " +
            codeToString(PI_ERROR_PROFILING_INFO_NOT_AVAILABLE));
  return MHostProfilingInfo->getStartTime();
}

template <>
uint64_t event_impl::get_profiling_info<info::event_profiling::command_end>(
    std::shared_ptr<ext::oneapi::experimental::detail::node_impl> NodeImpl) {
  if (isEventFromSubmitedExecCommandBuffer()) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Node profiling information is only available for "
                          "event returned from graph submission" +
                              codeToString(PI_ERROR_INVALID_VALUE));
  }
  checkProfilingPreconditions();
  // Node profiling is only available for Level-Zero backend
  if (MContext->getBackend() != backend::ext_oneapi_level_zero) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Profiling info is not available. " +
            codeToString(PI_ERROR_PROFILING_INFO_NOT_AVAILABLE));
  }
  if (!MHostEvent) {
    if (MEvent) {
      sycl::detail::pi::PiExtSyncPoint SyncPoint =
          MExecGraph->getSyncPointFromNode(NodeImpl);

      auto EndTime =
          get_event_profiling_info<info::event_profiling::command_end>(
              this->getHandleRef(), SyncPoint, this->getPlugin());
      if (!MFallbackProfiling) {
        return EndTime;
      } else {
        auto DeviceBaseTime =
            get_event_profiling_info<info::event_profiling::command_submit>(
                this->getHandleRef(), this->getPlugin());
        return MHostBaseTime - DeviceBaseTime + EndTime;
      }
    }
    return 0;
  }
  if (!MHostProfilingInfo)
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Profiling info is not available. " +
            codeToString(PI_ERROR_PROFILING_INFO_NOT_AVAILABLE));
  return MHostProfilingInfo->getEndTime();
}

template <> uint32_t event_impl::get_info<info::event::reference_count>() {
  if (!MHostEvent && MEvent) {
    return get_event_info<info::event::reference_count>(this->getHandleRef(),
                                                        this->getPlugin());
  }
  return 0;
}

template <>
info::event_command_status
event_impl::get_info<info::event::command_execution_status>() {
  if (MState == HES_Discarded)
    return info::event_command_status::ext_oneapi_unknown;

  if (!MHostEvent) {
    // Command is enqueued and PiEvent is ready
    if (MEvent)
      return get_event_info<info::event::command_execution_status>(
          this->getHandleRef(), this->getPlugin());
    // Command is blocked and not enqueued, PiEvent is not assigned yet
    else if (MCommand)
      return sycl::info::event_command_status::submitted;
  }

  return MHostEvent && MState.load() != HES_Complete
             ? sycl::info::event_command_status::submitted
             : info::event_command_status::complete;
}

void HostProfilingInfo::start() { StartTime = getTimestamp(); }

void HostProfilingInfo::end() { EndTime = getTimestamp(); }

pi_native_handle event_impl::getNative() {
  ensureContextInitialized();

  auto Plugin = getPlugin();
  if (!MIsInitialized) {
    MIsInitialized = true;
    auto TempContext = MContext.get()->getHandleRef();
    Plugin->call<PiApiKind::piEventCreate>(TempContext, &MEvent);
  }
  if (MContext->getBackend() == backend::opencl)
    Plugin->call<PiApiKind::piEventRetain>(getHandleRef());
  pi_native_handle Handle;
  Plugin->call<PiApiKind::piextEventGetNativeHandle>(getHandleRef(), &Handle);
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
  // Some events might not have a native handle underneath even at this point,
  // e.g. those produced by memset with 0 size (no PI call is made).
  if (MIsFlushed || !MEvent)
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
  pi_event_status Status = PI_EVENT_QUEUED;
  getPlugin()->call<PiApiKind::piEventGetInfo>(
      MEvent, PI_EVENT_INFO_COMMAND_EXECUTION_STATUS, sizeof(pi_int32), &Status,
      nullptr);
  if (Status == PI_EVENT_QUEUED) {
    getPlugin()->call<PiApiKind::piQueueFlush>(Queue->getHandleRef());
  }
  MIsFlushed = true;
}

void event_impl::cleanupDependencyEvents() {
  std::lock_guard<std::mutex> Lock(MMutex);
  MPreparedDepsEvents.clear();
  MPreparedHostDepsEvents.clear();
}

void event_impl::cleanDepEventsThroughOneLevel() {
  std::lock_guard<std::mutex> Lock(MMutex);
  for (auto &Event : MPreparedDepsEvents) {
    Event->cleanupDependencyEvents();
  }
  for (auto &Event : MPreparedHostDepsEvents) {
    Event->cleanupDependencyEvents();
  }
}

void event_impl::setSubmissionTime() {
  if (!MIsProfilingEnabled)
    return;
  if (!MFallbackProfiling) {
    if (QueueImplPtr Queue = MQueue.lock()) {
      try {
        MSubmitTime = Queue->getDeviceImplPtr()->getCurrentDeviceTime();
      } catch (sycl::exception &e) {
        if (e.code() == sycl::errc::feature_not_supported)
          throw sycl::exception(
              make_error_code(errc::profiling),
              std::string("Unable to get command group submission time: ") +
                  e.what());
        std::rethrow_exception(std::current_exception());
      }
    }
  } else {
    // Capture the host timestamp for a return value of function call
    // <info::event_profiling::command_submit>. See MFallbackProfiling
    MSubmitTime = getTimestamp();
  }
}

void event_impl::setHostEnqueueTime() {
  if (!MIsProfilingEnabled || !MFallbackProfiling)
    return;
  // Capture a host timestamp to use normalize profiling time in
  // <command_start> and <command_end>. See MFallbackProfiling
  MHostBaseTime = getTimestamp();
}

uint64_t event_impl::getSubmissionTime() { return MSubmitTime; }

bool event_impl::isCompleted() {
  return get_info<info::event::command_execution_status>() ==
         info::event_command_status::complete;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
