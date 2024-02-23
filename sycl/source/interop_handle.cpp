//==------------ interop_handle.cpp --- SYCL interop handle ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/accessor_impl.hpp>
#include <detail/backend_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/exception.hpp>
#include <sycl/interop_handle.hpp>

#include <algorithm>

namespace sycl {
inline namespace _V1 {

backend interop_handle::get_backend() const noexcept {
  return detail::getImplBackend(MQueue);
}

pi_native_handle interop_handle::getNativeMem(detail::Requirement *Req) const {
  auto Iter = std::find_if(std::begin(MMemObjs), std::end(MMemObjs),
                           [=](ReqToMem Elem) { return (Elem.first == Req); });

  if (Iter == std::end(MMemObjs)) {
    throw invalid_object_error("Invalid memory object used inside interop",
                               PI_ERROR_INVALID_MEM_OBJECT);
  }

  auto Plugin = MQueue->getPlugin();
  pi_native_handle Handle;
  Plugin->call<detail::PiApiKind::piextMemGetNativeHandle>(
      Iter->second, MDevice->getHandleRef(), &Handle);
  return Handle;
}

pi_native_handle interop_handle::getNativeDevice() const {
  return MDevice->getNative();
}

pi_native_handle interop_handle::getNativeContext() const {
  return MContext->getNative();
}

pi_native_handle
interop_handle::getNativeQueue(int32_t &NativeHandleDesc) const {
  return MQueue->getNative(NativeHandleDesc);
}

void interop_handle::addNativeEvents(
    std::vector<pi_native_handle> &NativeEvents) {
  auto Plugin = MQueue->getPlugin();

  if (!MEvent->backendSet()) {
    MEvent->setContextImpl(MContext);
  }

  // Make a std::vector of PiEvents from the native events
  for (auto i = 0; i < NativeEvents.size(); ++i) {
    detail::pi::PiEvent Ev;
    Plugin->call<detail::PiApiKind::piextEventCreateWithNativeHandle>(
        NativeEvents[i], MContext->getHandleRef(),
        /*OwnNativeHandle*/ true, &Ev);
    auto EventImpl = std::make_shared<detail::event_impl>(
        Ev, detail::createSyclObjFromImpl<context>(MContext));
    // TODO: Do I need to call things like:
    // setStateIncomplete -> Not sure
    // setSubmissionTime  -> Not sure
    // More...?
    MEvent->addHostTaskNativeEvent(EventImpl);
  }
}

std::vector<pi_native_handle> interop_handle::getNativeEvents() const {
  if (!MEvent->backendSet()) {
    MEvent->setContextImpl(MContext);
  }
  // What if the events here have not yet been enqueued? I will need to wait on
  // them. That is probably already done?
  //
  // Moreover what are the usual requirements of the host task launch?
  //
  // Do all dependent events need to be complete, or just enqueued? I suspect it
  // is the former, and we want the latter in the case that we are using these
  // entry points. We will maybe need a new host task entry point.
  std::vector<pi_native_handle> RetEvents;
  for (auto &DepEvent : MEvent->getWaitList()) {
    if (DepEvent->backendSet()) {
      auto NativeEvents = DepEvent->getNativeVector();
      RetEvents.insert(RetEvents.end(), NativeEvents.begin(),
                       NativeEvents.end());
    }
  }
  return RetEvents;
}

} // namespace _V1
} // namespace sycl
