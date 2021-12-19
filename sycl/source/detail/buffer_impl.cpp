//==----------------- buffer_impl.cpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/buffer_impl.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <detail/context_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include "xpti/xpti_trace_framework.hpp"
#include <detail/xpti_registry.hpp>
#include <sstream>
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
void *buffer_impl::allocateMem(ContextImplPtr Context, bool InitFromUserData,
                               void *HostPtr, RT::PiEvent &OutEventToWait) {
  bool HostPtrReadOnly = false;
  BaseT::determineHostPtr(Context, InitFromUserData, HostPtr, HostPtrReadOnly);

  assert(!(nullptr == HostPtr && BaseT::useHostPtr() && Context->is_host()) &&
         "Internal error. Allocating memory on the host "
         "while having use_host_ptr property");
  auto MemBuffer = MemoryManager::allocateMemBuffer(
      std::move(Context), this, HostPtr, HostPtrReadOnly, BaseT::getSize(),
      BaseT::MInteropEvent, BaseT::MInteropContext, MProps, OutEventToWait);
  associateNotification(MemBuffer);
  return MemBuffer;
}

void buffer_impl::constructorNotification(
    const detail::code_location &CodeLoc) {
  (void)CodeLoc;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  StreamID = xptiRegisterStream(SYCL_BUFFER_STREAM_NAME);

  // We try to create a unique string for the buffer constructor call by
  // combining it with the the created object address
  xpti::utils::StringHelper NG;
  std::string Name = NG.nameWithAddress<buffer_impl *>("buffer", this);
  xpti::offload_buffer_data_t BufConstr{(uintptr_t)this};

  xpti::payload_t Payload(
      Name.c_str(), (CodeLoc.fileName() ? CodeLoc.fileName() : ""),
      CodeLoc.lineNumber(), CodeLoc.columnNumber(), (void *)this);

  // constructor calls could be at different user-code locations; We create a
  // new event based on the code location info and if this has been seen before,
  // a previously created event will be returned.
  TraceEvent =
      xptiMakeEvent(Name.c_str(), &Payload, xpti::trace_offload_buffer_event,
                    xpti_at::active, &IId);
  IId = xptiGetUniqueId();
  xptiNotifySubscribers(StreamID, xpti::trace_offload_alloc_construct, nullptr,
                        TraceEvent, IId, &BufConstr);
#endif
}

void buffer_impl::associateNotification(void *MemObj) {
  (void)MemObj;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!(xptiTraceEnabled() && TraceEvent))
    return;
  xpti::offload_buffer_association_data_t BufAssoc{(uintptr_t)this,
                                                   (uintptr_t)MemObj};

  // Add assotiation between user level and PI level memory object
  xptiNotifySubscribers(StreamID, xpti::trace_offload_alloc_associate, nullptr,
                        TraceEvent, IId, &BufAssoc);
#endif
}

void buffer_impl::destructorNotification() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!(xptiTraceEnabled() && TraceEvent))
    return;
  // Destruction of user level memory object
  xptiNotifySubscribers(StreamID, xpti::trace_offload_alloc_destruct, nullptr,
                        TraceEvent, IId, nullptr);
#endif
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
