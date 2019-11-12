//==------------------ queue_impl.cpp - SYCL queue -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/clusm.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/usm_dispatch.hpp>
#include <CL/sycl/device.hpp>

#include <cstring>

namespace cl {
namespace sycl {
namespace detail {
template <> cl_uint queue_impl::get_info<info::queue::reference_count>() const {
  RT::PiResult result = PI_SUCCESS;
  if (!is_host())
    PI_CALL(RT::piQueueGetInfo, m_CommandQueue, PI_QUEUE_INFO_REFERENCE_COUNT,
            sizeof(result), &result, nullptr);
  return result;
}

template <> context queue_impl::get_info<info::queue::context>() const {
  return get_context();
}

template <> device queue_impl::get_info<info::queue::device>() const {
  return get_device();
}

event queue_impl::memset(std::shared_ptr<detail::queue_impl> Impl, void *Ptr,
                         int Value, size_t Count) {
  context Context = get_context();
  RT::PiEvent Event = nullptr;
  MemoryManager::fill_usm(Ptr, Impl, Count, Value, /*DepEvents*/ {}, Event);

  if (Context.is_host())
    return event();

  return event(pi::cast<cl_event>(Event), Context);
}

event queue_impl::memcpy(std::shared_ptr<detail::queue_impl> Impl, void *Dest,
                         const void *Src, size_t Count) {
  context Context = get_context();
  RT::PiEvent Event = nullptr;
  // Not entirely sure when UseExclusiveQueue should be true
  MemoryManager::copy_usm(Src, Impl, Count, Dest, /*DepEvents*/ {},
                          /*ExclusiveQueue*/ false, Event);

  if (Context.is_host())
    return event();

  return event(pi::cast<cl_event>(Event), Context);
}

event queue_impl::mem_advise(const void *Ptr, size_t Length, int Advice) {
  context Context = get_context();
  if (Context.is_host()) {
    return event();
  }

  // non-Host device
  std::shared_ptr<usm::USMDispatcher> USMDispatch =
      getSyclObjImpl(Context)->getUSMDispatch();
  RT::PiEvent Event = nullptr;

  USMDispatch->memAdvise(getHandleRef(), Ptr, Length, Advice, &Event);

  return event(pi::cast<cl_event>(Event), Context);
}
} // namespace detail
} // namespace sycl
} // namespace cl
