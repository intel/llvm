//==---------------- event_impl.cpp - SYCL event ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/event_info.hpp>
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
void event_impl::initContextIfNeeded() {
  if (MContext || !MIsDefaultConstructed)
    return;

  const device SyclDevice;
  MIsHostEvent = false;
  MContext =
      detail::queue_impl::getDefaultOrNew(*detail::getSyclObjImpl(SyclDevice));
  assert(MContext);
}

event_impl::~event_impl() {
  try {
    auto Handle = this->getHandle();
    if (Handle)
      getAdapter().call<UrApiKind::urEventRelease>(Handle);
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~event_impl", e);
  }
}

void event_impl::waitInternal(bool *Success) {
  auto Handle = this->getHandle();
  if (!MIsHostEvent && Handle) {
    // Wait for the native event
    ur_result_t Err =
        getAdapter().call_nocheck<UrApiKind::urEventWait>(1, &Handle);
    // TODO drop the UR_RESULT_ERROR_UKNOWN from here (this was waiting for
    // https://github.com/oneapi-src/unified-runtime/issues/1459 which is now
    // closed).
    if (Success != nullptr &&
        (Err == UR_RESULT_ERROR_UNKNOWN ||
         Err == UR_RESULT_ERROR_IN_EVENT_LIST_EXEC_STATUS))
      *Success = false;
    else {
      getAdapter().checkUrResult(Err);
      if (Success != nullptr)
        *Success = true;
    }
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
    Event->wait();
  for (const std::weak_ptr<event_impl> &WeakEventPtr :
       MWeakPostCompleteEvents) {
    if (EventImplPtr Event = WeakEventPtr.lock())
      Event->wait();
  }
}

void event_impl::setComplete() {
  if (MIsHostEvent || !this->getHandle()) {
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

static uint64_t inline getTimestamp(device_impl *Device) {
  if (Device) {
    try {
      return Device->getCurrentDeviceTime();
    } catch (sycl::exception &e) {
      if (e.code() == sycl::errc::feature_not_supported)
        throw sycl::exception(
            make_error_code(errc::profiling),
            std::string("Unable to get command group submission time: ") +
                e.what());
      std::rethrow_exception(std::current_exception());
    }
  } else {
    // Returning host time
    using namespace std::chrono;
    return duration_cast<nanoseconds>(
               high_resolution_clock::now().time_since_epoch())
        .count();
  }
}

ur_event_handle_t event_impl::getHandle() const { return MEvent.load(); }

void event_impl::setHandle(const ur_event_handle_t &UREvent) {
  MEvent.store(UREvent);
}

context_impl &event_impl::getContextImpl() {
  initContextIfNeeded();
  assert(MContext && "Trying to get context from a host event!");
  return *MContext;
}

adapter_impl &event_impl::getAdapter() {
  initContextIfNeeded();
  return MContext->getAdapter();
}

void event_impl::setStateIncomplete() { MState = HES_NotComplete; }

void event_impl::setContextImpl(context_impl &Context) {
  MIsHostEvent = false;
  MContext = Context.shared_from_this();
}

event_impl::event_impl(ur_event_handle_t Event, const context &SyclContext,
                       private_tag)
    : MEvent(Event), MContext(detail::getSyclObjImpl(SyclContext)),
      MIsFlushed(true), MState(HES_Complete) {

  ur_context_handle_t TempContext;
  getAdapter().call<UrApiKind::urEventGetInfo>(
      this->getHandle(), UR_EVENT_INFO_CONTEXT, sizeof(ur_context_handle_t),
      &TempContext, nullptr);

  if (MContext->getHandleRef() != TempContext) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "The syclContext must match the OpenCL context "
                          "associated with the clEvent. " +
                              codeToString(UR_RESULT_ERROR_INVALID_CONTEXT));
  }
}

event_impl::event_impl(queue_impl &Queue, private_tag)
    : MQueue{Queue.weak_from_this()},
      MIsProfilingEnabled{Queue.MIsProfilingEnabled} {
  this->setContextImpl(Queue.getContextImpl());
  MState.store(HES_Complete);
}

event_impl::event_impl(HostEventState State, private_tag) : MState(State) {
  MIsHostEvent = true;
  if (State == HES_Discarded || State == HES_Complete)
    MIsFlushed = true;
}

void event_impl::setQueue(queue_impl &Queue) {
  MQueue = Queue.weak_from_this();
  MIsProfilingEnabled = Queue.MIsProfilingEnabled;

  // TODO After setting the queue, the event is no longer default
  // constructed. Consider a design change which would allow
  // for such a change regardless of the construction method.
  MIsDefaultConstructed = false;
}

void event_impl::initHostProfilingInfo() {
  assert(isHost() && "This must be a host event");
  assert(MState == HES_NotComplete &&
         "Host event must be incomplete to initialize profiling info");

  std::shared_ptr<queue_impl> QueuePtr = MSubmittedQueue.lock();
  assert(QueuePtr && "Queue must be valid to initialize host profiling info");
  assert(QueuePtr->MIsProfilingEnabled && "Queue must have profiling enabled");

  MIsProfilingEnabled = true;
  MHostProfilingInfo = std::make_unique<HostProfilingInfo>();
  device_impl &Device = QueuePtr->getDeviceImpl();
  MHostProfilingInfo->setDevice(&Device);
}

void event_impl::setSubmittedQueue(std::weak_ptr<queue_impl> SubmittedQueue) {
  MSubmittedQueue = std::move(SubmittedQueue);
}

#ifdef XPTI_ENABLE_INSTRUMENTATION
void *event_impl::instrumentationProlog(std::string &Name,
                                        xpti::stream_id_t StreamID,
                                        uint64_t &IId) const {
  void *TraceEvent = nullptr;
  constexpr uint16_t NotificationTraceType = xpti::trace_wait_begin;
  if (!xptiCheckTraceEnabled(StreamID, NotificationTraceType))
    return TraceEvent;
  xpti::trace_event_data_t *WaitEvent = nullptr;

  // Create a string with the event address so it
  // can be associated with other debug data
  xpti::utils::StringHelper SH;
  Name = SH.nameWithAddress<ur_event_handle_t>("event.wait", this->getHandle());

  // We can emit the wait associated with the graph if the
  // event does not have a command object or associated with
  // the command object, if it exists
  if (MCommand) {
    Command *Cmd = (Command *)MCommand;
    WaitEvent = Cmd->MTraceEvent ? static_cast<xpti_td *>(Cmd->MTraceEvent)
                                 : GSYCLGraphEvent;
  } else {
    // If queue.wait() is used, we want to make sure the information about the
    // queue is available with the wait events. We check to see if the
    // TraceEvent is available in the Queue object.
    void *TraceEvent = nullptr;
    if (std::shared_ptr<queue_impl> Queue = MQueue.lock()) {
      TraceEvent = Queue->getTraceEvent();
      WaitEvent =
          (TraceEvent ? static_cast<xpti_td *>(TraceEvent) : GSYCLGraphEvent);
    } else
      WaitEvent = GSYCLGraphEvent;
  }
  // Record the current instance ID for use by Epilog
  IId = xptiGetUniqueId();
  xptiNotifySubscribers(StreamID, NotificationTraceType, nullptr, WaitEvent,
                        IId, static_cast<const void *>(Name.c_str()));
  TraceEvent = (void *)WaitEvent;
  return TraceEvent;
}

void event_impl::instrumentationEpilog(void *TelemetryEvent,
                                       const std::string &Name,
                                       xpti::stream_id_t StreamID,
                                       uint64_t IId) const {
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

void event_impl::wait(bool *Success) {
  if (MState == HES_Discarded)
    throw sycl::exception(make_error_code(errc::invalid),
                          "wait method cannot be used for a discarded event.");

  if (!MGraph.expired()) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "wait method cannot be used for an event associated "
                          "with a command graph.");
  }

#ifdef XPTI_ENABLE_INSTRUMENTATION
  void *TelemetryEvent = nullptr;
  uint64_t IId = 0;
  std::string Name;
  xpti::stream_id_t StreamID = xptiRegisterStream(SYCL_STREAM_NAME);
  TelemetryEvent = instrumentationProlog(Name, StreamID, IId);
#endif

  auto EventHandle = getHandle();
  if (EventHandle)
    // presence of the native handle means the command has been enqueued, so no
    // need to go via the slow path event waiting in the scheduler
    waitInternal(Success);
  else if (MCommand)
    detail::Scheduler::getInstance().waitForEvent(*this, Success);

#ifdef XPTI_ENABLE_INSTRUMENTATION
  instrumentationEpilog(TelemetryEvent, Name, StreamID, IId);
#endif
}

void event_impl::wait_and_throw() {
  wait();

  if (std::shared_ptr<queue_impl> SubmittedQueue = MSubmittedQueue.lock())
    SubmittedQueue->throw_asynchronous();
}

void event_impl::checkProfilingPreconditions() const {
  std::weak_ptr<queue_impl> EmptyPtr;

  if (!MIsHostEvent && !EmptyPtr.owner_before(MQueue) &&
      !MQueue.owner_before(EmptyPtr)) {
    throw sycl::exception(make_error_code(sycl::errc::invalid),
                          "Profiling information is unavailable as the event "
                          "has no associated queue.");
  }
  if (!MIsProfilingEnabled && !MProfilingTagEvent) {
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
  if (isProfilingTagEvent()) {
    // For profiling tag events we rely on the submission time reported as
    // the start time has undefined behavior.
    return get_event_profiling_info<info::event_profiling::command_submit>(
        this->getHandle(), this->getAdapter());
  }

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
  auto Handle = this->getHandle();
  if (MEventFromSubmittedExecCommandBuffer && !MIsHostEvent && Handle) {
    uint64_t StartTime =
        get_event_profiling_info<info::event_profiling::command_start>(
            Handle, this->getAdapter());
    if (StartTime < MSubmitTime)
      MSubmitTime = StartTime;
  }
  return MSubmitTime;
}

template <>
uint64_t
event_impl::get_profiling_info<info::event_profiling::command_start>() {
  checkProfilingPreconditions();
  if (!MIsHostEvent) {
    auto Handle = getHandle();
    if (Handle) {
      return get_event_profiling_info<info::event_profiling::command_start>(
          Handle, this->getAdapter());
    }
    // If command is nop (for example, USM operations for 0 bytes) return
    // recorded submission time. If event is created using default constructor,
    // 0 will be returned.
    return MSubmitTime;
  }
  if (!MHostProfilingInfo)
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Profiling info is not available. " +
            codeToString(UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE));
  return MHostProfilingInfo->getStartTime();
}

template <>
uint64_t event_impl::get_profiling_info<info::event_profiling::command_end>() {
  checkProfilingPreconditions();
  if (!MIsHostEvent) {
    auto Handle = this->getHandle();
    if (Handle) {
      return get_event_profiling_info<info::event_profiling::command_end>(
          Handle, this->getAdapter());
    }
    // If command is nop (for example, USM operations for 0 bytes) return
    // recorded submission time. If event is created using default constructor,
    // 0 will be returned.
    return MSubmitTime;
  }
  if (!MHostProfilingInfo)
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Profiling info is not available. " +
            codeToString(UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE));
  return MHostProfilingInfo->getEndTime();
}

template <> uint32_t event_impl::get_info<info::event::reference_count>() {
  auto Handle = this->getHandle();
  if (!MIsHostEvent && Handle) {
    return get_event_info<info::event::reference_count>(Handle,
                                                        this->getAdapter());
  }
  return 0;
}

template <>
info::event_command_status
event_impl::get_info<info::event::command_execution_status>() {
  if (MState == HES_Discarded)
    return info::event_command_status::ext_oneapi_unknown;

  if (!MIsHostEvent) {
    // Command is enqueued and UrEvent is ready
    auto Handle = this->getHandle();
    if (Handle)
      return get_event_info<info::event::command_execution_status>(
          Handle, this->getAdapter());
    // Command is blocked and not enqueued, UrEvent is not assigned yet
    else if (MCommand)
      return sycl::info::event_command_status::submitted;
  }

  return MIsHostEvent && MState.load() != HES_Complete
             ? sycl::info::event_command_status::submitted
             : info::event_command_status::complete;
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
template <>
typename info::platform::version::return_type
event_impl::get_backend_info<info::platform::version>() const {
  if (!MContext) {
    return "Context not initialized, no backend info available";
  }
  if (MContext->getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::platform::version info descriptor can "
                          "only be queried with an OpenCL backend");
  }
  if (std::shared_ptr<queue_impl> Queue = MQueue.lock()) {
    return Queue->getDeviceImpl()
        .get_platform()
        .get_info<info::platform::version>();
  }
  // If the queue has been released, no platform will be associated
  // so return empty string.
  return "";
}
#endif

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
template <>
typename info::device::version::return_type
event_impl::get_backend_info<info::device::version>() const {
  if (!MContext) {
    return "Context not initialized, no backend info available";
  }
  if (MContext->getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::device::version info descriptor can only "
                          "be queried with an OpenCL backend");
  }
  if (std::shared_ptr<queue_impl> Queue = MQueue.lock()) {
    return Queue->getDeviceImpl().get_info<info::device::version>();
  }
  return ""; // If the queue has been released, no device will be associated so
             // return empty string
}
#endif

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
template <>
typename info::device::backend_version::return_type
event_impl::get_backend_info<info::device::backend_version>() const {
  if (!MContext) {
    return "Context not initialized, no backend info available";
  }
  if (MContext->getBackend() != backend::ext_oneapi_level_zero) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::device::backend_version info descriptor "
                          "can only be queried with a Level Zero backend");
  }
  return "";
  // Currently The Level Zero backend does not define the value of this
  // information descriptor and implementations are encouraged to return the
  // empty string as per specification.
}
#endif

void HostProfilingInfo::start() { StartTime = getTimestamp(Device); }

void HostProfilingInfo::end() { EndTime = getTimestamp(Device); }

ur_native_handle_t event_impl::getNative() {
  if (isHost())
    return {};
  initContextIfNeeded();

  adapter_impl &Adapter = getAdapter();
  auto Handle = getHandle();
  if (MIsDefaultConstructed && !Handle) {
    auto TempContext = MContext.get()->getHandleRef();
    ur_event_native_properties_t NativeProperties{};
    ur_event_handle_t UREvent = nullptr;
    Adapter.call<UrApiKind::urEventCreateWithNativeHandle>(
        0u, TempContext, &NativeProperties, &UREvent);
    this->setHandle(UREvent);
    Handle = UREvent;
  }
  ur_native_handle_t OutHandle;
  Adapter.call<UrApiKind::urEventGetNativeHandle>(Handle, &OutHandle);
  if (MContext->getBackend() == backend::opencl)
    __SYCL_OCL_CALL(clRetainEvent, ur::cast<cl_event>(OutHandle));
  return OutHandle;
}

std::vector<EventImplPtr> event_impl::getWaitList() {
  std::lock_guard<std::mutex> Lock(MMutex);

  std::vector<EventImplPtr> Result;
  Result.reserve(MPreparedDepsEvents.size() + MPreparedHostDepsEvents.size());
  Result.insert(Result.end(), MPreparedDepsEvents.begin(),
                MPreparedDepsEvents.end());
  Result.insert(Result.end(), MPreparedHostDepsEvents.begin(),
                MPreparedHostDepsEvents.end());

  return Result;
}

void event_impl::flushIfNeeded(queue_impl *UserQueue) {
  // Some events might not have a native handle underneath even at this point,
  // e.g. those produced by memset with 0 size (no UR call is made).
  auto Handle = this->getHandle();
  if (MIsFlushed || !Handle)
    return;

  std::shared_ptr<queue_impl> Queue = MQueue.lock();
  // If the queue has been released, all of the commands have already been
  // implicitly flushed by urQueueRelease.
  if (!Queue) {
    MIsFlushed = true;
    return;
  }
  if (Queue.get() == UserQueue)
    return;

  // Check if the task for this event has already been submitted.
  ur_event_status_t Status = UR_EVENT_STATUS_QUEUED;
  getAdapter().call<UrApiKind::urEventGetInfo>(
      Handle, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS, sizeof(ur_event_status_t),
      &Status, nullptr);
  if (Status == UR_EVENT_STATUS_QUEUED) {
    getAdapter().call<UrApiKind::urQueueFlush>(Queue->getHandleRef());
  }
  MIsFlushed = true;
}

void event_impl::cleanupDependencyEvents() {
  std::lock_guard<std::mutex> Lock(MMutex);
  MPreparedDepsEvents.clear();
  MPreparedHostDepsEvents.clear();
}

void event_impl::cleanDepEventsThroughOneLevelUnlocked() {
  for (auto &Event : MPreparedDepsEvents) {
    Event->cleanupDependencyEvents();
  }
  for (auto &Event : MPreparedHostDepsEvents) {
    Event->cleanupDependencyEvents();
  }
}

void event_impl::cleanDepEventsThroughOneLevel() {
  std::lock_guard<std::mutex> Lock(MMutex);
  cleanDepEventsThroughOneLevelUnlocked();
}

void event_impl::setSubmissionTime() {
  if (!MIsProfilingEnabled && !MProfilingTagEvent)
    return;

  if (std::shared_ptr<queue_impl> Queue =
          isHost() ? MSubmittedQueue.lock() : MQueue.lock()) {
    device_impl &Device = Queue->getDeviceImpl();
    MSubmitTime = getTimestamp(&Device);
  }
}

uint64_t event_impl::getSubmissionTime() { return MSubmitTime; }

bool event_impl::isCompleted() {
  return get_info<info::event::command_execution_status>() ==
         info::event_command_status::complete;
}

void event_impl::setCommand(Command *Cmd) { MCommand = Cmd; }

} // namespace detail
} // namespace _V1
} // namespace sycl
