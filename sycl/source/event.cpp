//==---------------- event.cpp --- SYCL event ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/backend_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/event.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/stl.hpp>

#include <memory>
#include <unordered_set>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

event::event() : impl(std::make_shared<detail::event_impl>(std::nullopt)) {}

event::event(cl_event ClEvent, const context &SyclContext)
    : impl(std::make_shared<detail::event_impl>(
          detail::pi::cast<RT::PiEvent>(ClEvent), SyclContext)) {
  // This is a special interop constructor for OpenCL, so the event must be
  // retained.
  impl->getPlugin().call<detail::PiApiKind::piEventRetain>(
      detail::pi::cast<RT::PiEvent>(ClEvent));
}

bool event::operator==(const event &rhs) const { return rhs.impl == impl; }

bool event::operator!=(const event &rhs) const { return !(*this == rhs); }

cl_event event::get() const { return impl->get(); }

bool event::is_host() const { return impl->is_host(); }

void event::wait() { impl->wait(impl); }

void event::wait(const std::vector<event> &EventList) {
  for (auto E : EventList) {
    E.wait();
  }
}

void event::wait_and_throw() { impl->wait_and_throw(impl); }

void event::wait_and_throw(const std::vector<event> &EventList) {
  for (auto E : EventList) {
    E.wait_and_throw();
  }
}

std::vector<event> event::get_wait_list() {
  std::vector<event> Result;

  for (auto &EventImpl : impl->getWaitList())
    Result.push_back(detail::createSyclObjFromImpl<event>(EventImpl));

  return Result;
}

event::event(std::shared_ptr<detail::event_impl> event_impl)
    : impl(event_impl) {}

template <typename Param>
typename detail::is_event_info_desc<Param>::return_type
event::get_info() const {
  return impl->template get_info<Param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template __SYCL_EXPORT ReturnT event::get_info<info::event::Desc>() const;

#include <sycl/info/event_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, PiCode)              \
  template <>                                                                  \
  __SYCL_EXPORT ReturnT event::get_profiling_info<info::DescType::Desc>()      \
      const {                                                                  \
    impl->wait(impl);                                                          \
    return impl->get_profiling_info<info::DescType::Desc>();                   \
  }

#include <sycl/info/event_profiling_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

backend event::get_backend() const noexcept { return getImplBackend(impl); }

pi_native_handle event::getNative() const { return impl->getNative(); }

std::vector<pi_native_handle> event::getNativeVector() const {
  std::vector<pi_native_handle> ReturnVector = {impl->getNative()};
  return ReturnVector;
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
