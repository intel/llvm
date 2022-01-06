//==---------- xpti_registry.cpp ----- XPTI Stream Registry ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/global_handler.hpp>
#include <detail/xpti_registry.hpp>

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include "xpti/xpti_trace_framework.hpp"
#include <sstream>
#endif
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
xpti::trace_event_data_t *
XPTIRegistry::createTraceEvent(void *Obj, const char *ObjName, uint64_t &IId,
                               const detail::code_location &CodeLoc,
                               uint16_t TraceEventType) {
  std::string Name;
  if (CodeLoc.fileName()) {
    Name = std::string(CodeLoc.fileName()) + ":" +
           std::to_string(CodeLoc.lineNumber()) + ":" +
           std::to_string(CodeLoc.columnNumber());
  } else {
    xpti::utils::StringHelper NG;
    Name = NG.nameWithAddress<void *>(ObjName, Obj);
  }
  xpti::payload_t Payload(
      Name.c_str(), (CodeLoc.fileName() ? CodeLoc.fileName() : ""),
      CodeLoc.lineNumber(), CodeLoc.columnNumber(), (void *)Obj);

  // Calls could be at different user-code locations; We create a new event
  // based on the code location info and if this has been seen before, a
  // previously created event will be returned.
  return xptiMakeEvent(Name.c_str(), &Payload, TraceEventType, xpti_at::active,
                       &IId);
}
void XPTIRegistry::bufferConstructorNotification(
    void *UserObj, const detail::code_location &CodeLoc) {
  (void)UserObj;
  (void)CodeLoc;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  GlobalHandler::instance().getXPTIRegistry().initializeFrameworkOnce();
  if (!xptiTraceEnabled())
    return;

  uint64_t IId;
  xpti::offload_buffer_data_t BufConstr{(uintptr_t)UserObj};

  xpti::trace_event_data_t *TraceEvent = createTraceEvent(
      UserObj, "buffer", IId, CodeLoc, xpti::trace_offload_buffer_event);
  xptiNotifySubscribers(GBufferStreamID, xpti::trace_offload_alloc_construct,
                        nullptr, TraceEvent, IId, &BufConstr);
#endif
}

void XPTIRegistry::bufferAssociateNotification(void *UserObj, void *MemObj) {
  (void)UserObj;
  (void)MemObj;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  uint64_t IId = xptiGetUniqueId();
  xpti::offload_buffer_association_data_t BufAssoc{(uintptr_t)UserObj,
                                                   (uintptr_t)MemObj};

  // Add association between user level and PI level memory object
  xptiNotifySubscribers(GBufferStreamID, xpti::trace_offload_alloc_associate,
                        nullptr, nullptr, IId, &BufAssoc);
#endif
}

void XPTIRegistry::bufferReleaseNotification(void *UserObj, void *MemObj) {
  (void)UserObj;
  (void)MemObj;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  uint64_t IId = xptiGetUniqueId();
  xpti::offload_buffer_association_data_t BufRelease{(uintptr_t)UserObj,
                                                     (uintptr_t)MemObj};

  // Release PI level memory object
  xptiNotifySubscribers(GBufferStreamID, xpti::trace_offload_alloc_release,
                        nullptr, nullptr, IId, &BufRelease);
#endif
}

void XPTIRegistry::bufferDestructorNotification(void *UserObj) {
  (void)UserObj;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;
  uint64_t IId = xptiGetUniqueId();
  xpti::offload_buffer_data_t BufDestr{(uintptr_t)UserObj};
  // Destruction of user level memory object
  xptiNotifySubscribers(GBufferStreamID, xpti::trace_offload_alloc_destruct,
                        nullptr, nullptr, IId, &BufDestr);
#endif
}

void XPTIRegistry::bufferAccessorNotification(
    void *UserObj, void *AccessorObj, uint32_t Target, uint32_t Mode,
    const detail::code_location &CodeLoc) {
  (void)UserObj;
  (void)AccessorObj;
  (void)CodeLoc;
  (void)Target;
  (void)Mode;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;

  uint64_t IId;
  xpti::offload_accessor_data_t AccessorConstr{
      (uintptr_t)UserObj, (uintptr_t)AccessorObj, Target, Mode};

  xpti::trace_event_data_t *TraceEvent = createTraceEvent(
      UserObj, "accessor", IId, CodeLoc, xpti::trace_offload_accessor_event);
  xptiNotifySubscribers(GBufferStreamID, xpti::trace_offload_alloc_accessor,
                        nullptr, TraceEvent, IId, &AccessorConstr);
#endif
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
