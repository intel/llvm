//==------- reusable_events.cpp --- SYCL reusable events -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "detail/context_impl.hpp"
#include "detail/event_impl.hpp"
#include "detail/queue_impl.hpp"
#include <detail/sycl_mem_obj_t.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <typename PropertyListT = empty_properties_t>
sycl::event make_event(const sycl::context &ctxt, PropertyListT props) {
  ur_exp_event_desc_t Desc = {};
  std::vector<sycl::device> &ContextDevices = ctxt.get_devices();
  detail::context_impl &ContextImpl = *sycl::detail::getSyclObjImpl(ctxt);
  sycl::detail::adapter_impl &Adapter = ContextImpl.getAdapter();
  ur_event_handle_t EventHandle = nullptr;

  if (ContextDevices.size() == 0) {
    // TODO exception
  }

  Desc.stype = UR_STRUCTURE_TYPE_EXP_EVENT_DESC;
  Desc.hDevice = detail::getSyclObjImpl(ContextDevices[0])->getHandleRef();
  // TODO Desc.flags

  ur_result_t Result =
      Adapter.call_nocheck<sycl::detail::UrApiKind::urEventCreateExp>(
          ContextImpl.getHandleRef(), &Desc, &EventHandle);
  if (Result != UR_RESULT_SUCCESS) {
    throw sycl::exception(sycl::make_error_code(errc::runtime),
                          "Failed to create an event.");
  }

  auto ResEvent = detail::event_impl::create_from_handle(EventHandle, ctxt);
  ResEvent->setStateIncomplete();

  return detail::createSyclObjFromImpl<sycl::event>(std::move(ResEvent));
}

void enqueue_wait_event(sycl::queue q, const event& evt) {
	q.ext_oneapi_submit_barrier({evt});
}

void enqueue_wait_events(sycl::queue q, const std::vector<event>& evts) {
	q.ext_oneapi_submit_barrier(evts);
}

void enqueue_signal_event(sycl::queue q, event& evt) {
  detail::getSyclObjImpl(q)->submit_barrier_direct_without_event(
      {}, detail::CGType::Barrier, detail::code_location::current(),
      detail::getSyclObjImpl(evt));
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
