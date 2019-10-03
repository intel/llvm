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

namespace cl {
namespace sycl {
namespace detail {

// Threat all devices that don't support interoperability as host devices to
// avoid attempts to call method get on such events.
bool event_impl::is_host() const { return m_HostEvent || !m_OpenCLInterop; }

cl_event event_impl::get() const {
  if (m_OpenCLInterop) {
    PI_CALL(RT::piEventRetain(m_Event));
    return pi::cast<cl_event>(m_Event);
  }
  throw invalid_object_error(
      "This instance of event doesn't support OpenCL interoperability.");
}

event_impl::~event_impl() {
  if (m_Event) {
    PI_CALL(RT::piEventRelease(m_Event));
  }
}

void event_impl::setComplete() {
  PI_CALL(RT::piEventSetStatus(m_Event, CL_COMPLETE));
}

void event_impl::waitInternal() const {
  if (!m_HostEvent) {
    PI_CALL(RT::piEventsWait(1, &m_Event));
  }
  // Waiting of host events is NOP so far as all operations on host device
  // are blocking.
}

const RT::PiEvent &event_impl::getHandleRef() const { return m_Event; }
RT::PiEvent &event_impl::getHandleRef() { return m_Event; }

const ContextImplPtr &event_impl::getContextImpl() { return m_Context; }

void event_impl::setContextImpl(const ContextImplPtr &Context) {
  m_HostEvent = Context->is_host();
  m_OpenCLInterop = !m_HostEvent;
  m_Context = Context;
}

event_impl::event_impl(cl_event CLEvent, const context &SyclContext)
    : m_Context(detail::getSyclObjImpl(SyclContext)),
      m_OpenCLInterop(true), m_HostEvent(false) {

  m_Event = pi::cast<RT::PiEvent>(CLEvent);

  if (m_Context->is_host()) {
    throw cl::sycl::invalid_parameter_error(
        "The syclContext must match the OpenCL context associated with the "
        "clEvent.");
  }

  RT::PiContext TempContext;
  PI_CALL(RT::piEventGetInfo(
      m_Event, CL_EVENT_CONTEXT, sizeof(RT::PiContext), &TempContext, nullptr));
  if (m_Context->getHandleRef() != TempContext) {
    throw cl::sycl::invalid_parameter_error(
        "The syclContext must match the OpenCL context associated with the "
        "clEvent.");
  }

  PI_CALL(RT::piEventRetain(m_Event));
}

event_impl::event_impl(std::shared_ptr<cl::sycl::detail::queue_impl> Queue) {
  if (Queue->is_host() &&
      Queue->has_property<property::queue::enable_profiling>()) {
    m_HostProfilingInfo.reset(new HostProfilingInfo());
    if (!m_HostProfilingInfo)
      throw runtime_error("Out of host memory");
  }
}

void event_impl::wait(
    std::shared_ptr<cl::sycl::detail::event_impl> Self) const {

  if (m_Event || m_HostEvent)
    // presence of m_Event means the command has been enqueued, so no need to
    // go via the slow path event waiting in the scheduler
    waitInternal();
  else
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
}

template <>
cl_ulong
event_impl::get_profiling_info<info::event_profiling::command_submit>() const {
  if (!m_HostEvent) {
    return get_event_profiling_info<
        info::event_profiling::command_submit>::_(this->getHandleRef());
  }
  if (!m_HostProfilingInfo)
    throw invalid_object_error("Profiling info is not available.");
  return m_HostProfilingInfo->getStartTime();
}

template <>
cl_ulong
event_impl::get_profiling_info<info::event_profiling::command_start>() const {
  if (!m_HostEvent) {
    return get_event_profiling_info<info::event_profiling::command_start>::_(
        this->getHandleRef());
  }
  if (!m_HostProfilingInfo)
    throw invalid_object_error("Profiling info is not available.");
  return m_HostProfilingInfo->getStartTime();
}

template <>
cl_ulong
event_impl::get_profiling_info<info::event_profiling::command_end>() const {
  if (!m_HostEvent) {
    return get_event_profiling_info<info::event_profiling::command_end>::_(
        this->getHandleRef());
  }
  if (!m_HostProfilingInfo)
    throw invalid_object_error("Profiling info is not available.");
  return m_HostProfilingInfo->getEndTime();
}

template <> cl_uint event_impl::get_info<info::event::reference_count>() const {
  if (!m_HostEvent) {
    return get_event_info<info::event::reference_count>::_(
        this->getHandleRef());
  }
  return 0;
}

template <>
info::event_command_status
event_impl::get_info<info::event::command_execution_status>() const {
  if (!m_HostEvent) {
    return get_event_info<info::event::command_execution_status>::_(
        this->getHandleRef());
  }
  return info::event_command_status::complete;
}

static uint64_t getTimestamp() {
  auto ts = std::chrono::high_resolution_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(ts).count();
}

void HostProfilingInfo::start() { StartTime = getTimestamp(); }

void HostProfilingInfo::end() { EndTime = getTimestamp(); }

} // namespace detail
} // namespace sycl
} // namespace cl
