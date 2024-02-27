//==---------------- helpers.cpp - SYCL helpers ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/scheduler/commands.hpp>
#include <sycl/detail/helpers.hpp>

#include <detail/buffer_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/event.hpp>

#include <memory>

namespace sycl {
inline namespace _V1 {
using ContextImplPtr = std::shared_ptr<sycl::detail::context_impl>;
namespace detail {
// TODO: remove from public header files and implementation during the next ABI
// Breaking window. Not used any more.
std::vector<sycl::detail::pi::PiEvent>
getOrWaitEvents(std::vector<sycl::event> DepEvents, ContextImplPtr Context) {
  std::vector<sycl::detail::pi::PiEvent> Events;
  for (auto SyclEvent : DepEvents) {
    auto SyclEventImplPtr = detail::getSyclObjImpl(SyclEvent);
    // throwaway events created with empty constructor will not have a context
    // (which is set lazily) calling getContextImpl() would set that
    // context, which we wish to avoid as it is expensive.
    if (!SyclEventImplPtr->isContextInitialized() &&
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
      // In this path nullptr native event means that the command has not been
      // enqueued. It may happen if async enqueue in a host task is involved.
      // This should affect only shortcut functions, which bypass the graph.
      if (SyclEventImplPtr->getHandleRef() == nullptr) {
        std::vector<Command *> AuxCmds;
        Scheduler::getInstance().enqueueCommandForCG(SyclEventImplPtr, AuxCmds,
                                                     BLOCKING);
      }
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

void markBufferAsInternal(const std::shared_ptr<buffer_impl> &BufImpl) {
  BufImpl->markAsInternal();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
