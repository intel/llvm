//==------ timeline_semaphore.cpp --- SYCL timeline semaphores -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/timeline_semaphore_impl.hpp>
#include <sycl/ext/oneapi/experimental/timeline_semaphore.hpp>

#include <string>

namespace sycl {
inline namespace _V1 {
namespace detail {

timeline_semaphore_impl::timeline_semaphore_impl(
    const std::shared_ptr<context_impl> &Context, uint64_t InitialValue)
    : MContext(Context) {
  checkValue(InitialValue);
  MContext->getAdapter().call<UrApiKind::urTimelineSemaphoreCreateExp>(
      MContext->getHandleRef(), InitialValue, &MSemaphore);
}

timeline_semaphore_impl::~timeline_semaphore_impl() {
  try {
    MContext->getAdapter().call<UrApiKind::urTimelineSemaphoreReleaseExp>(
        MSemaphore);
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~timeline_semaphore_impl",
                                      e);
  }
}

context timeline_semaphore_impl::get_context() const {
  return createSyclObjFromImpl<context>(MContext);
}

void timeline_semaphore_impl::checkValue(context_impl &Context,
                                         uint64_t Value) {
  const uint64_t MaxValue = Context.getTimelineSemaphoreMaxValue();
  if (Value > MaxValue)
    throw sycl::exception(make_error_code(errc::invalid),
                          "The value " + std::to_string(Value) +
                              " exceeds the maximum value " +
                              std::to_string(MaxValue) +
                              " of a timeline semaphore in this context.");
}

void timeline_semaphore_impl::wait(uint64_t Value) const {
  checkValue(Value);
  MContext->getAdapter().call<UrApiKind::urTimelineSemaphoreHostWaitExp>(
      MSemaphore, Value);
}

void timeline_semaphore_impl::signal(uint64_t Value) const {
  checkValue(Value);
  MContext->getAdapter().call<UrApiKind::urTimelineSemaphoreHostSignalExp>(
      MSemaphore, Value);
}

ur_result_t timeline_semaphore_impl::enqueue(
    queue_impl &Queue, uint64_t Value, bool IsSignal,
    const std::vector<ur_event_handle_t> &RawEvents,
    ur_event_handle_t *OutEvent) const {
  adapter_impl &Adapter = Queue.getAdapter();
  auto EnqueueOp = [&](uint32_t NumEvents, const ur_event_handle_t *Events,
                       ur_event_handle_t *Event) {
    return IsSignal ? Adapter.call_nocheck<
                          UrApiKind::urEnqueueTimelineSemaphoreSignalExp>(
                          Queue.getHandleRef(), MSemaphore, Value, NumEvents,
                          Events, Event)
                    : Adapter.call_nocheck<
                          UrApiKind::urEnqueueTimelineSemaphoreWaitExp>(
                          Queue.getHandleRef(), MSemaphore, Value, NumEvents,
                          Events, Event);
  };

  if (Queue.isInOrder())
    return EnqueueOp(RawEvents.size(), RawEvents.data(), OutEvent);

  // A barrier without a wait list waits for all previously enqueued commands,
  // so the explicit dependencies are waited for in a separate command first.
  if (!RawEvents.empty()) {
    if (ur_result_t Res = Adapter.call_nocheck<UrApiKind::urEnqueueEventsWait>(
            Queue.getHandleRef(), RawEvents.size(), RawEvents.data(), nullptr);
        Res != UR_RESULT_SUCCESS)
      return Res;
  }
  if (ur_result_t Res =
          Adapter.call_nocheck<UrApiKind::urEnqueueEventsWaitWithBarrierExt>(
              Queue.getHandleRef(), nullptr, 0, nullptr, nullptr);
      Res != UR_RESULT_SUCCESS)
    return Res;

  ur_event_handle_t OpEvent = nullptr;
  if (ur_result_t Res = EnqueueOp(0, nullptr, &OpEvent);
      Res != UR_RESULT_SUCCESS)
    return Res;

  ur_result_t Res =
      Adapter.call_nocheck<UrApiKind::urEnqueueEventsWaitWithBarrierExt>(
          Queue.getHandleRef(), nullptr, 1, &OpEvent, OutEvent);
  Adapter.call_nocheck<UrApiKind::urEventRelease>(OpEvent);
  return Res;
}

} // namespace detail

namespace ext::oneapi::experimental {

void timeline_semaphore::create(const context &ctxt, uint64_t initial_value) {
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(ctxt);
  if (!ContextImpl->supportsTimelineSemaphores())
    throw sycl::exception(sycl::make_error_code(errc::feature_not_supported),
                          "Not all devices in the context support "
                          "aspect::ext_oneapi_timeline_semaphore.");

  impl = std::make_shared<sycl::detail::timeline_semaphore_impl>(ContextImpl,
                                                                 initial_value);
}

context timeline_semaphore::get_context() const { return impl->get_context(); }

void timeline_semaphore::wait(uint64_t value) const { impl->wait(value); }

void timeline_semaphore::signal(uint64_t value) { impl->signal(value); }

static void
checkQueueForSemaphoreOperation(sycl::detail::queue_impl &QueueImpl,
                                sycl::detail::timeline_semaphore_impl &SemImpl,
                                const char *FuncName) {
  if (&QueueImpl.getContextImpl() != &SemImpl.getContextImpl())
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          std::string(FuncName) +
                              ": the queue and the timeline semaphore must "
                              "have the same context.");

  if (QueueImpl.hasCommandGraph() ||
      (QueueImpl.getContextImpl().isNativeRecordingActive() &&
       QueueImpl.isNativeRecording()))
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          std::string(FuncName) +
                              " is not supported on a queue which is "
                              "recording a graph.");
}

void enqueue_signal(queue q, timeline_semaphore &sem, uint64_t value,
                    const sycl::detail::code_location &CodeLoc) {
  sycl::detail::queue_impl &QueueImpl = *sycl::detail::getSyclObjImpl(q);
  auto SemImpl = sycl::detail::getSyclObjImpl(sem);
  checkQueueForSemaphoreOperation(QueueImpl, *SemImpl, "enqueue_signal");
  SemImpl->checkValue(value);

  QueueImpl.submit_timeline_semaphore_direct(SemImpl, value,
                                             /*IsSignal*/ true, CodeLoc);
}

void enqueue_wait(queue q, const timeline_semaphore &sem, uint64_t value,
                  const sycl::detail::code_location &CodeLoc) {
  sycl::detail::queue_impl &QueueImpl = *sycl::detail::getSyclObjImpl(q);
  auto SemImpl = sycl::detail::getSyclObjImpl(sem);
  checkQueueForSemaphoreOperation(QueueImpl, *SemImpl, "enqueue_wait");
  SemImpl->checkValue(value);

  QueueImpl.submit_timeline_semaphore_direct(SemImpl, value,
                                             /*IsSignal*/ false, CodeLoc);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
