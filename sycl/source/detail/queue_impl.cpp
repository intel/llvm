//==------------------ queue_impl.cpp - SYCL queue -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/clusm.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/usm_dispatch.hpp>
#include <CL/sycl/device.hpp>

namespace cl {
namespace sycl {
namespace detail {
template <> cl_uint queue_impl::get_info<info::queue::reference_count>() const {
  RT::PiResult result = PI_SUCCESS;
  if (!is_host())
    PI_CALL(RT::piQueueGetInfo(m_CommandQueue,
                               PI_QUEUE_INFO_REFERENCE_COUNT,
                               sizeof(result), &result, nullptr));
  return result;
}

template <> context queue_impl::get_info<info::queue::context>() const {
  return get_context();
}

template <> device queue_impl::get_info<info::queue::device>() const {
  return get_device();
}

event queue_impl::memset(void *Ptr, int Value, size_t Count) {
  context Context = get_context();
  std::shared_ptr<usm::USMDispatcher> USMDispatch =
      getSyclObjImpl(Context)->getUSMDispatch();
  cl_event Event;

  PI_CHECK(USMDispatch->enqueueMemset(getHandleRef(), Ptr, Value, Count,
                                      /* sizeof waitlist */ 0, nullptr,
                                      reinterpret_cast<pi_event *>(&Event)));

  return event(Event, Context);
}

event queue_impl::memcpy(void *Dest, const void *Src, size_t Count) {
  context Context = get_context();
  std::shared_ptr<usm::USMDispatcher> USMDispatch =
      getSyclObjImpl(Context)->getUSMDispatch();
  cl_event Event;

  PI_CHECK(USMDispatch->enqueueMemcpy(getHandleRef(),
                                      /* blocking */ false, Dest, Src, Count,
                                      /* sizeof waitlist */ 0, nullptr,
                                      reinterpret_cast<pi_event *>(&Event)));

  return event(Event, Context);
}

event queue_impl::mem_advise(const void *Ptr, size_t Length, int Advice) {
  context Context = get_context();
  std::shared_ptr<usm::USMDispatcher> USMDispatch =
    getSyclObjImpl(Context)->getUSMDispatch();
  cl_event Event;

  USMDispatch->memAdvise(getHandleRef(), Ptr, Length, Advice,
                         reinterpret_cast<pi_event *>(&Event));

  return event(Event, Context);
}
} // namespace detail
} // namespace sycl
} // namespace cl
