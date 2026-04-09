//==------- reusable_events.cpp --- SYCL reusable events -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/detail/ur.hpp>
#include "detail/event_impl.hpp"
#include "detail/queue_impl.hpp"
#include <detail/sycl_mem_obj_t.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template<typename PropertyListT = empty_properties_t>
sycl::event make_event(const sycl::device &dev, const sycl::context &ctxt, PropertyListT props = {}) {

}

void enqueue_wait_event(sycl::queue q, const event& evt) {
	q.ext_oneapi_submit_barrier({evt});
}

void enqueue_wait_events(sycl::queue q, const std::vector<event>& evts) {
	q.ext_oneapi_submit_barrier(evts);
}

void enqueue_signal_event(sycl::queue q, event& evt) {
	// q.ext_oneapi_enqueue_signal_event(evt);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
