//==---------------- helpers.cpp - SYCL helpers ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/scheduler/commands.hpp>
#include <sycl/detail/helpers.hpp>

#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <sycl/event.hpp>

#include <memory>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
using ContextImplPtr = std::shared_ptr<sycl::detail::context_impl>;
namespace detail {
std::vector<RT::PiEvent> getOrWaitEvents(std::vector<sycl::event> DepEvents,
                                         ContextImplPtr Context) {
  std::vector<RT::PiEvent> Events;
  for (auto SyclEvent : DepEvents) {
    auto SyclEventImplPtr = detail::getSyclObjImpl(SyclEvent);
    // throwaway events created with empty constructor will not have a context
    // (which is set lazily) calling getContextImpl() would set that
    // context, which we wish to avoid as it is expensive.
    if (SyclEventImplPtr->MIsContextInitialized == false &&
        !SyclEventImplPtr->is_host()) {
      continue;
    }
    // The fusion command and its event are associated with a non-host context,
    // but still does not produce a PI event.
    bool NoPiEvent =
        SyclEventImplPtr->MCommand &&
        !static_cast<Command *>(SyclEventImplPtr->MCommand)->producesPiEvent();
    if (SyclEventImplPtr->is_host() ||
        SyclEventImplPtr->getContextImpl() != Context || NoPiEvent) {
      // Call wait, because the command for the event might not have been
      // enqueued when kernel fusion is happening.
      SyclEventImplPtr->wait(SyclEventImplPtr);
    } else {
      Events.push_back(SyclEventImplPtr->getHandleRef());
    }
  }
  return Events;
}

void waitEvents(std::vector<sycl::event> DepEvents) {
  for (auto SyclEvent : DepEvents) {
    detail::getSyclObjImpl(SyclEvent)->waitInternal();
  }
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
