//==---------------- event.cpp --- SYCL event ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/event.hpp>

#include <CL/sycl/stl.hpp>

#include <memory>
#include <unordered_set>

__SYCL_INLINE namespace cl {
namespace sycl {

event::event() : impl(std::make_shared<detail::event_impl>()) {}

event::event(cl_event ClEvent, const context &SyclContext)
    : impl(std::make_shared<detail::event_impl>(
          detail::pi::cast<RT::PiEvent>(ClEvent), SyclContext)) {}

bool event::operator==(const event &rhs) const { return rhs.impl == impl; }

bool event::operator!=(const event &rhs) const { return !(*this == rhs); }

cl_event event::get() { return impl->get(); }

bool event::is_host() const { return impl->is_host(); }

void event::wait() { impl->wait(impl); }

void event::wait(const vector_class<event> &EventList) {
  for (auto E : EventList) {
    E.wait();
  }
}

void event::wait_and_throw() { impl->wait_and_throw(impl); }

void event::wait_and_throw(const vector_class<event> &EventList) {
  for (auto E : EventList) {
    E.wait_and_throw();
  }
}

vector_class<event> event::get_wait_list() {
  vector_class<event> Result;

  for (auto &EventImpl : detail::Scheduler::getInstance().getWaitList(impl))
    Result.push_back(detail::createSyclObjFromImpl<event>(EventImpl));

  return Result;
}

event::event(shared_ptr_class<detail::event_impl> event_impl)
    : impl(event_impl) {}

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
    template <> ret_type event::get_info<info::param_type::param>() const {    \
      return impl->get_info<info::param_type::param>();                        \
    }

#include <CL/sycl/info/event_traits.def>

#undef PARAM_TRAITS_SPEC

#define PARAM_TRAITS_SPEC(param_type, param, ret_type)                         \
    template <>                                                                \
    ret_type event::get_profiling_info<info::param_type::param>() const {      \
      impl->wait(impl);                                                        \
      return impl->get_profiling_info<info::param_type::param>();              \
    }

#include <CL/sycl/info/event_profiling_traits.def>

#undef PARAM_TRAITS_SPEC

} // namespace sycl
} // namespace cl
