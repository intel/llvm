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
#ifdef XPTI_ENABLE_INSTRUMENTATION
xpti::trace_event_data_t *XPTIRegistry::createTraceEvent(
    const void *Obj, const void *FuncPtr, uint64_t &IId,
    const detail::code_location &CodeLoc, uint16_t TraceEventType) {
  xpti::utils::StringHelper NG;
  auto Name = NG.nameWithAddress<void *>(CodeLoc.functionName(),
                                         const_cast<void *>(FuncPtr));
  xpti::payload_t Payload(Name.c_str(),
                          (CodeLoc.fileName() ? CodeLoc.fileName() : ""),
                          CodeLoc.lineNumber(), CodeLoc.columnNumber(), Obj);

  // Calls could be at different user-code locations; We create a new event
  // based on the code location info and if this has been seen before, a
  // previously created event will be returned.
  return xptiMakeEvent(Name.c_str(), &Payload, TraceEventType, xpti_at::active,
                       &IId);
}
#endif // XPTI_ENABLE_INSTRUMENTATION

void XPTIRegistry::bufferConstructorNotification(
    const void *UserObj, const detail::code_location &CodeLoc,
    const void *HostObj, const void *Type, uint32_t Dim, uint32_t ElemSize,
    size_t Range[3]) {
  (void)UserObj;
  (void)CodeLoc;
  (void)HostObj;
  (void)Type;
  (void)Dim;
  (void)ElemSize;
  (void)Range;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  GlobalHandler::instance().getXPTIRegistry().initializeFrameworkOnce();
  if (!xptiTraceEnabled())
    return;

  uint64_t IId;
  xpti::offload_buffer_data_t BufConstr{(uintptr_t)UserObj,
                                        (uintptr_t)HostObj,
                                        (const char *)Type,
                                        ElemSize,
                                        Dim,
                                        {Range[0], Range[1], Range[2]}};

  xpti::trace_event_data_t *TraceEvent = createTraceEvent(
      UserObj, "buffer", IId, CodeLoc, xpti::trace_offload_buffer_event);
  xptiNotifySubscribers(GBufferStreamID, xpti::trace_offload_alloc_construct,
                        nullptr, TraceEvent, IId, &BufConstr);
#endif
}

void XPTIRegistry::bufferAssociateNotification(const void *UserObj,
                                               const void *MemObj) {
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

void XPTIRegistry::bufferReleaseNotification(const void *UserObj,
                                             const void *MemObj) {
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

void XPTIRegistry::bufferDestructorNotification(const void *UserObj) {
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
    const void *UserObj, const void *AccessorObj, uint32_t Target,
    uint32_t Mode, const detail::code_location &CodeLoc) {
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

void XPTIRegistry::kernelEnqueueNotification(
    const void *Kernel, NDRDescT &NDRDesc, std::vector<ArgDesc> &Args,
    const detail::code_location &CodeLoc) {
  (void)Kernel;
  (void)NDRDesc;
  (void)Args;
  (void)CodeLoc;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;

  uint64_t IId;

  xpti::trace_event_data_t *TraceEvent = createTraceEvent(
      Kernel, "kernel", IId, CodeLoc, xpti::trace_offload_kernel_enqueue_event);
  xpti::offload_kernel_enqueue_data_t KernelData{
      {NDRDesc.GlobalSize[0], NDRDesc.GlobalSize[1], NDRDesc.GlobalSize[2]},
      {NDRDesc.LocalSize[0], NDRDesc.LocalSize[1], NDRDesc.LocalSize[2]},
      {NDRDesc.GlobalOffset[0], NDRDesc.GlobalOffset[1],
       NDRDesc.GlobalOffset[2]},
      Args.size()};
  for (size_t i = 0; i < Args.size(); i++) {
    std::string Prefix("arg");
    xpti::offload_kernel_arg_data_t arg{(int)Args[i].MType, Args[i].MPtr,
                                        Args[i].MSize, Args[i].MIndex};
    xpti::addMetadata(TraceEvent, Prefix + std::to_string(i), arg);
  }

  xptiNotifySubscribers(GBufferStreamID, xpti::trace_offload_kernel_enqueue,
                        nullptr, TraceEvent, IId, &KernelData);
#endif
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
