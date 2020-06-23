//==------------------ queue_impl.cpp - SYCL queue -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/device.hpp>
#include <detail/queue_impl.hpp>

#include <cstring>

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include "xpti_trace_framework.hpp"
#include <sstream>
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
template <> cl_uint queue_impl::get_info<info::queue::reference_count>() const {
  RT::PiResult result = PI_SUCCESS;
  if (!is_host())
    getPlugin().call<PiApiKind::piQueueGetInfo>(
        MCommandQueue, PI_QUEUE_INFO_REFERENCE_COUNT, sizeof(result), &result,
        nullptr);
  return result;
}

template <> context queue_impl::get_info<info::queue::context>() const {
  return get_context();
}

template <> device queue_impl::get_info<info::queue::device>() const {
  return get_device();
}

event queue_impl::memset(shared_ptr_class<detail::queue_impl> Impl, void *Ptr,
                         int Value, size_t Count) {
  context Context = get_context();
  RT::PiEvent Event = nullptr;
  MemoryManager::fill_usm(Ptr, Impl, Count, Value, /*DepEvents*/ {}, Event);

  if (Context.is_host())
    return event();

  event ResEvent{pi::cast<cl_event>(Event), Context};
  addUSMEvent(ResEvent);
  return ResEvent;
}

event queue_impl::memcpy(shared_ptr_class<detail::queue_impl> Impl, void *Dest,
                         const void *Src, size_t Count) {
  context Context = get_context();
  RT::PiEvent Event = nullptr;
  MemoryManager::copy_usm(Src, Impl, Count, Dest, /*DepEvents*/ {}, Event);

  if (Context.is_host())
    return event();

  event ResEvent{pi::cast<cl_event>(Event), Context};
  addUSMEvent(ResEvent);
  return ResEvent;
}

event queue_impl::mem_advise(const void *Ptr, size_t Length,
                             pi_mem_advice Advice) {
  context Context = get_context();
  if (Context.is_host()) {
    return event();
  }

  // non-Host device
  RT::PiEvent Event = nullptr;
  const detail::plugin &Plugin = getPlugin();
  Plugin.call<PiApiKind::piextUSMEnqueueMemAdvise>(getHandleRef(), Ptr, Length,
                                                   Advice, &Event);

  event ResEvent{pi::cast<cl_event>(Event), Context};
  addUSMEvent(ResEvent);
  return ResEvent;
}

void queue_impl::addEvent(event Event) {
  std::weak_ptr<event_impl> EventWeakPtr{getSyclObjImpl(Event)};
  std::lock_guard<mutex_class> Guard(MMutex);
  MEvents.push_back(std::move(EventWeakPtr));
}

void queue_impl::addUSMEvent(event Event) {
  std::lock_guard<mutex_class> Guard(MMutex);
  MUSMEvents.push_back(std::move(Event));
}

void *queue_impl::instrumentationProlog(const detail::code_location &CodeLoc,
                                        string_class &Name, int32_t StreamID,
                                        uint64_t &IId) {
  void *TraceEvent = nullptr;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  xpti::trace_event_data_t *WaitEvent = nullptr;
  if (!xptiTraceEnabled())
    return TraceEvent;

  xpti::payload_t Payload;
  bool HasSourceInfo = false;
  // We try to create a unique string for the wait() call by combining it with
  // the queue address
  xpti::utils::StringHelper NG;
  Name = NG.nameWithAddress<queue_impl *>("queue.wait", this);

  if (!CodeLoc.fileName()) {
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
  WaitEvent = xptiMakeEvent(Name.c_str(), &Payload, xpti::trace_graph_event,
                            xpti_at::active, &QWaitInstanceNo);
  IId = QWaitInstanceNo;
  if (WaitEvent) {
    device D = get_device();
    std::string DevStr;
    if (D.is_host())
      DevStr = "HOST";
    else if (D.is_cpu())
      DevStr = "CPU";
    else if (D.is_gpu())
      DevStr = "GPU";
    else if (D.is_accelerator())
      DevStr = "ACCELERATOR";
    else
      DevStr = "UNKNOWN";
    xptiAddMetadata(WaitEvent, "sycl_device", DevStr.c_str());
    if (HasSourceInfo) {
      xptiAddMetadata(WaitEvent, "sym_function_name", CodeLoc.functionName());
      xptiAddMetadata(WaitEvent, "sym_source_file_name", CodeLoc.fileName());
      xptiAddMetadata(WaitEvent, "sym_line_no",
                      std::to_string(CodeLoc.lineNumber()).c_str());
    }
    xptiNotifySubscribers(StreamID, xpti::trace_wait_begin, nullptr, WaitEvent,
                          QWaitInstanceNo,
                          static_cast<const void *>(Name.c_str()));
    TraceEvent = (void *)WaitEvent;
  }
#endif
  return TraceEvent;
}

void queue_impl::instrumentationEpilog(void *TelemetryEvent, string_class &Name,
                                       int32_t StreamID, uint64_t IId) {
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

void queue_impl::wait(const detail::code_location &CodeLoc) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  void *TelemetryEvent = nullptr;
  uint64_t IId;
  std::string Name;
  int32_t StreamID = xptiRegisterStream(SYCL_STREAM_NAME);
  TelemetryEvent = instrumentationProlog(CodeLoc, Name, StreamID, IId);
#endif

  std::lock_guard<mutex_class> Guard(MMutex);
  for (std::weak_ptr<event_impl> &EventImplWeakPtr : MEvents) {
    if (std::shared_ptr<event_impl> EventImplPtr = EventImplWeakPtr.lock())
      EventImplPtr->wait(EventImplPtr);
  }
  for (event &Event : MUSMEvents) {
    Event.wait();
  }
  MEvents.clear();

#ifdef XPTI_ENABLE_INSTRUMENTATION
  instrumentationEpilog(TelemetryEvent, Name, StreamID, IId);
#endif
}

void queue_impl::initHostTaskAndEventCallbackThreadPool() {
  if (MHostTaskThreadPool)
    return;

  int Size = 1;

  if (const char *val = std::getenv("SYCL_QUEUE_THREAD_POOL_SIZE"))
    try {
      Size = std::stoi(val);
    } catch (...) {
      throw invalid_parameter_error(
          "Invalid value for SYCL_QUEUE_THREAD_POOL_SIZE environment variable",
          PI_INVALID_VALUE);
    }

  if (Size < 1)
    throw invalid_parameter_error(
        "Invalid value for SYCL_QUEUE_THREAD_POOL_SIZE environment variable",
        PI_INVALID_VALUE);

  MHostTaskThreadPool.reset(new ThreadPool(Size));
  MHostTaskThreadPool->start();
}

pi_native_handle queue_impl::getNative() const {
  auto Plugin = getPlugin();
  pi_native_handle Handle;
  Plugin.call<PiApiKind::piextQueueGetNativeHandle>(MCommandQueue, &Handle);
  return Handle;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
