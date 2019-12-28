//==---------------- event_impl.cpp - SYCL event ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>

#include <chrono>

__SYCL_INLINE namespace cl {
namespace sycl {
namespace detail {

// Threat all devices that don't support interoperability as host devices to
// avoid attempts to call method get on such events.
bool event_impl::is_host() const { return MHostEvent || !MOpenCLInterop; }

cl_event event_impl::get() const {
  if (MOpenCLInterop) {
    PI_CALL(piEventRetain)(MEvent);
    return pi::cast<cl_event>(MEvent);
  }
  throw invalid_object_error(
      "This instance of event doesn't support OpenCL interoperability.");
}

event_impl::~event_impl() {
  if (MEvent)
    PI_CALL(piEventRelease)(MEvent);
}

void event_impl::waitInternal() const {
  if (!MHostEvent) {
    PI_CALL(piEventsWait)(1, &MEvent);
  }
  // Waiting of host events is NOP so far as all operations on host device
  // are blocking.
}

const RT::PiEvent &event_impl::getHandleRef() const { return MEvent; }
RT::PiEvent &event_impl::getHandleRef() { return MEvent; }

const ContextImplPtr &event_impl::getContextImpl() { return MContext; }

void event_impl::setContextImpl(const ContextImplPtr &Context) {
  MHostEvent = Context->is_host();
  MOpenCLInterop = !MHostEvent;
  MContext = Context;
}

event_impl::event_impl(RT::PiEvent Event, const context &SyclContext)
    : MEvent(Event), MContext(detail::getSyclObjImpl(SyclContext)),
      MOpenCLInterop(true), MHostEvent(false) {

  if (MContext->is_host()) {
    throw cl::sycl::invalid_parameter_error(
        "The syclContext must match the OpenCL context associated with the "
        "clEvent.");
  }

  RT::PiContext TempContext;
  PI_CALL(piEventGetInfo)(MEvent, CL_EVENT_CONTEXT, sizeof(RT::PiContext),
                          &TempContext, nullptr);
  if (MContext->getHandleRef() != TempContext) {
    throw cl::sycl::invalid_parameter_error(
        "The syclContext must match the OpenCL context associated with the "
        "clEvent.");
  }

  PI_CALL(piEventRetain)(MEvent);
}

event_impl::event_impl(QueueImplPtr Queue) : MQueue(Queue) {
  if (Queue->is_host() &&
      Queue->has_property<property::queue::enable_profiling>()) {
    MHostProfilingInfo.reset(new HostProfilingInfo());
    if (!MHostProfilingInfo)
      throw runtime_error("Out of host memory");
  }
}

void event_impl::wait(
    std::shared_ptr<cl::sycl::detail::event_impl> Self) const {

  if (MEvent)
    // presence of MEvent means the command has been enqueued, so no need to
    // go via the slow path event waiting in the scheduler
    waitInternal();
  else if (MCommand)
    detail::Scheduler::getInstance().waitForEvent(std::move(Self));
}

void event_impl::wait_and_throw(
    std::shared_ptr<cl::sycl::detail::event_impl> Self) {
  wait(Self);
  for (auto &EventImpl :
       detail::Scheduler::getInstance().getWaitList(std::move(Self))) {
    Command *Cmd = (Command *)EventImpl->getCommand();
    if (Cmd)
      Cmd->getQueue()->throw_asynchronous();
  }
  if (MQueue)
    MQueue->throw_asynchronous();
}

template <>
cl_ulong
event_impl::get_profiling_info<info::event_profiling::command_submit>() const {
  if (!MHostEvent) {
    return get_event_profiling_info<info::event_profiling::command_submit>::get(
        this->getHandleRef());
  }
  if (!MHostProfilingInfo)
    throw invalid_object_error("Profiling info is not available.");
  return MHostProfilingInfo->getStartTime();
}

template <>
cl_ulong
event_impl::get_profiling_info<info::event_profiling::command_start>() const {
  if (!MHostEvent) {
    return get_event_profiling_info<info::event_profiling::command_start>::get(
        this->getHandleRef());
  }
  if (!MHostProfilingInfo)
    throw invalid_object_error("Profiling info is not available.");
  return MHostProfilingInfo->getStartTime();
}

template <>
cl_ulong
event_impl::get_profiling_info<info::event_profiling::command_end>() const {
  if (!MHostEvent) {
    return get_event_profiling_info<info::event_profiling::command_end>::get(
        this->getHandleRef());
  }
  if (!MHostProfilingInfo)
    throw invalid_object_error("Profiling info is not available.");
  return MHostProfilingInfo->getEndTime();
}

template <> cl_uint event_impl::get_info<info::event::reference_count>() const {
  if (!MHostEvent) {
    return get_event_info<info::event::reference_count>::get(
        this->getHandleRef());
  }
  return 0;
}

template <>
info::event_command_status
event_impl::get_info<info::event::command_execution_status>() const {
  if (!MHostEvent) {
    return get_event_info<info::event::command_execution_status>::get(
        this->getHandleRef());
  }
  return info::event_command_status::complete;
}

static uint64_t getTimestamp() {
  auto TimeStamp = std::chrono::high_resolution_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(TimeStamp)
      .count();
}

void HostProfilingInfo::start() { StartTime = getTimestamp(); }

void HostProfilingInfo::end() { EndTime = getTimestamp(); }

} // namespace detail
} // namespace sycl
} // namespace cl
