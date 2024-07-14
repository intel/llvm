//==---------- xpti_registry.cpp ----- XPTI Stream Registry ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/global_handler.hpp>
#include <detail/queue_impl.hpp>
#include <detail/xpti_registry.hpp>

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include "xpti/xpti_trace_framework.hpp"
#include <sstream>
#endif
namespace sycl {
inline namespace _V1 {
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
  constexpr uint16_t NotificationTraceType =
      xpti::trace_offload_alloc_memory_object_construct;
  if (!xptiCheckTraceEnabled(GBufferStreamID, NotificationTraceType))
    return;

  uint64_t IId;
  xpti::offload_buffer_data_t BufConstr{(uintptr_t)UserObj,
                                        (uintptr_t)HostObj,
                                        (const char *)Type,
                                        ElemSize,
                                        Dim,
                                        {Range[0], Range[1], Range[2]}};

  xpti::trace_event_data_t *TraceEvent = createTraceEvent(
      UserObj, "buffer", IId, CodeLoc, xpti::trace_offload_memory_object_event);
  xptiNotifySubscribers(GBufferStreamID, NotificationTraceType, nullptr,
                        TraceEvent, IId, &BufConstr);
#endif
}

void XPTIRegistry::bufferAssociateNotification(const void *UserObj,
                                               const void *MemObj) {
  (void)UserObj;
  (void)MemObj;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      xpti::trace_offload_alloc_memory_object_associate;
  if (!xptiCheckTraceEnabled(GBufferStreamID, NotificationTraceType))
    return;
  uint64_t IId = xptiGetUniqueId();
  xpti::offload_association_data_t BufAssoc{(uintptr_t)UserObj,
                                            (uintptr_t)MemObj};

  // Add association between user level and PI level memory object
  xptiNotifySubscribers(GBufferStreamID, NotificationTraceType, nullptr,
                        nullptr, IId, &BufAssoc);
#endif
}

void XPTIRegistry::bufferReleaseNotification(const void *UserObj,
                                             const void *MemObj) {
  (void)UserObj;
  (void)MemObj;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      xpti::trace_offload_alloc_memory_object_release;
  if (!xptiCheckTraceEnabled(GBufferStreamID, NotificationTraceType))
    return;
  uint64_t IId = xptiGetUniqueId();
  xpti::offload_association_data_t BufRelease{(uintptr_t)UserObj,
                                              (uintptr_t)MemObj};

  // Release PI level memory object
  xptiNotifySubscribers(GBufferStreamID, NotificationTraceType, nullptr,
                        nullptr, IId, &BufRelease);
#endif
}

void XPTIRegistry::bufferDestructorNotification(const void *UserObj) {
  (void)UserObj;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      xpti::trace_offload_alloc_memory_object_destruct;
  if (!xptiCheckTraceEnabled(GBufferStreamID, NotificationTraceType))
    return;
  uint64_t IId = xptiGetUniqueId();
  xpti::offload_buffer_data_t BufDestr{(uintptr_t)UserObj};
  // Destruction of user level memory object
  xptiNotifySubscribers(GBufferStreamID, NotificationTraceType, nullptr,
                        nullptr, IId, &BufDestr);
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
  constexpr uint16_t NotificationTraceType = xpti::trace_offload_alloc_accessor;
  if (!xptiCheckTraceEnabled(GBufferStreamID, NotificationTraceType))
    return;

  uint64_t IId;
  xpti::offload_accessor_data_t AccessorConstr{
      (uintptr_t)UserObj, (uintptr_t)AccessorObj, Target, Mode};

  xpti::trace_event_data_t *TraceEvent = createTraceEvent(
      UserObj, "accessor", IId, CodeLoc, xpti::trace_offload_accessor_event);
  xptiNotifySubscribers(GBufferStreamID, NotificationTraceType, nullptr,
                        TraceEvent, IId, &AccessorConstr);
#endif
}

void XPTIRegistry::sampledImageConstructorNotification(
    const void *UserObj, const detail::code_location &CodeLoc,
    const void *HostObj, uint32_t Dim, size_t Range[3], uint32_t ImageFormat,
    uint32_t SamplerAddressingMode, uint32_t SamplerCoordinateNormalizationMode,
    uint32_t SamplerFilteringMode) {
  (void)UserObj;
  (void)CodeLoc;
  (void)HostObj;
  (void)Dim;
  (void)Range;
  (void)ImageFormat;
  (void)SamplerAddressingMode;
  (void)SamplerCoordinateNormalizationMode;
  (void)SamplerFilteringMode;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  GlobalHandler::instance().getXPTIRegistry().initializeFrameworkOnce();
  constexpr uint16_t NotificationTraceType =
      xpti::trace_offload_alloc_memory_object_construct;
  if (!xptiCheckTraceEnabled(GImageStreamID, NotificationTraceType))
    return;

  uint64_t IId;
  xpti::offload_image_data_t ImgConstr{(uintptr_t)UserObj,
                                       (uintptr_t)HostObj,
                                       Dim,
                                       {Range[0], Range[1], Range[2]},
                                       ImageFormat,
                                       SamplerAddressingMode,
                                       SamplerCoordinateNormalizationMode,
                                       SamplerFilteringMode};

  xpti::trace_event_data_t *TraceEvent =
      createTraceEvent(UserObj, "sampled_image", IId, CodeLoc,
                       xpti::trace_offload_memory_object_event);
  xptiNotifySubscribers(GImageStreamID, NotificationTraceType, nullptr,
                        TraceEvent, IId, &ImgConstr);
#endif
}

void XPTIRegistry::sampledImageDestructorNotification(const void *UserObj) {
  (void)UserObj;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      xpti::trace_offload_alloc_memory_object_destruct;
  if (!xptiCheckTraceEnabled(GImageStreamID, NotificationTraceType))
    return;
  uint64_t IId = xptiGetUniqueId();
  xpti::offload_image_data_t ImgDestr{(uintptr_t)UserObj};
  // Destruction of user level memory object
  xptiNotifySubscribers(GImageStreamID, NotificationTraceType, nullptr, nullptr,
                        IId, &ImgDestr);
#endif
}

void XPTIRegistry::unsampledImageConstructorNotification(
    const void *UserObj, const detail::code_location &CodeLoc,
    const void *HostObj, uint32_t Dim, size_t Range[3], uint32_t ImageFormat) {
  (void)UserObj;
  (void)CodeLoc;
  (void)HostObj;
  (void)Dim;
  (void)Range;
  (void)ImageFormat;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  GlobalHandler::instance().getXPTIRegistry().initializeFrameworkOnce();
  constexpr uint16_t NotificationTraceType =
      xpti::trace_offload_alloc_memory_object_construct;
  if (!xptiCheckTraceEnabled(GImageStreamID, NotificationTraceType))
    return;

  uint64_t IId;
  xpti::offload_image_data_t ImgConstr{(uintptr_t)UserObj,
                                       (uintptr_t)HostObj,
                                       Dim,
                                       {Range[0], Range[1], Range[2]},
                                       ImageFormat,
                                       // No sampler information
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt};

  xpti::trace_event_data_t *TraceEvent =
      createTraceEvent(UserObj, "unsampled_image", IId, CodeLoc,
                       xpti::trace_offload_memory_object_event);
  xptiNotifySubscribers(GImageStreamID, NotificationTraceType, nullptr,
                        TraceEvent, IId, &ImgConstr);
#endif
}

void XPTIRegistry::unsampledImageDestructorNotification(const void *UserObj) {
  (void)UserObj;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType =
      xpti::trace_offload_alloc_memory_object_destruct;
  if (!xptiCheckTraceEnabled(GImageStreamID, NotificationTraceType))
    return;
  uint64_t IId = xptiGetUniqueId();
  xpti::offload_image_data_t ImgDestr{(uintptr_t)UserObj};
  // Destruction of user level memory object
  xptiNotifySubscribers(GImageStreamID, NotificationTraceType, nullptr, nullptr,
                        IId, &ImgDestr);
#endif
}

void XPTIRegistry::unsampledImageAccessorNotification(
    const void *UserObj, const void *AccessorObj, uint32_t Target,
    uint32_t Mode, const void *Type, uint32_t ElemSize,
    const detail::code_location &CodeLoc) {
  (void)UserObj;
  (void)AccessorObj;
  (void)CodeLoc;
  (void)Target;
  (void)Mode;
  (void)Type;
  (void)ElemSize;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType = xpti::trace_offload_alloc_accessor;
  if (!xptiCheckTraceEnabled(GImageStreamID, NotificationTraceType))
    return;

  uint64_t IId;
  xpti::offload_image_accessor_data_t AccessorConstr{(uintptr_t)UserObj,
                                                     (uintptr_t)AccessorObj,
                                                     Target,
                                                     Mode,
                                                     (const char *)Type,
                                                     ElemSize};

  xpti::trace_event_data_t *TraceEvent =
      createTraceEvent(UserObj, "unsampled_image_accessor", IId, CodeLoc,
                       xpti::trace_offload_accessor_event);
  xptiNotifySubscribers(GImageStreamID, NotificationTraceType, nullptr,
                        TraceEvent, IId, &AccessorConstr);
#endif
}

void XPTIRegistry::unsampledImageHostAccessorNotification(
    const void *UserObj, const void *AccessorObj, uint32_t Mode,
    const void *Type, uint32_t ElemSize, const detail::code_location &CodeLoc) {
  (void)UserObj;
  (void)AccessorObj;
  (void)CodeLoc;
  (void)Mode;
  (void)Type;
  (void)ElemSize;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType = xpti::trace_offload_alloc_accessor;
  if (!xptiCheckTraceEnabled(GImageStreamID, NotificationTraceType))
    return;

  uint64_t IId;
  xpti::offload_image_accessor_data_t AccessorConstr{
      (uintptr_t)UserObj, (uintptr_t)AccessorObj,
      std::nullopt,       Mode,
      (const char *)Type, ElemSize};

  xpti::trace_event_data_t *TraceEvent =
      createTraceEvent(UserObj, "host_unsampled_image_accessor", IId, CodeLoc,
                       xpti::trace_offload_accessor_event);
  xptiNotifySubscribers(GImageStreamID, NotificationTraceType, nullptr,
                        TraceEvent, IId, &AccessorConstr);
#endif
}

void XPTIRegistry::sampledImageAccessorNotification(
    const void *UserObj, const void *AccessorObj, uint32_t Target,
    const void *Type, uint32_t ElemSize, const detail::code_location &CodeLoc) {
  (void)UserObj;
  (void)AccessorObj;
  (void)CodeLoc;
  (void)Target;
  (void)Type;
  (void)ElemSize;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType = xpti::trace_offload_alloc_accessor;
  if (!xptiCheckTraceEnabled(GImageStreamID, NotificationTraceType))
    return;

  uint64_t IId;
  xpti::offload_image_accessor_data_t AccessorConstr{
      (uintptr_t)UserObj, (uintptr_t)AccessorObj, Target,
      std::nullopt,       (const char *)Type,     ElemSize};

  xpti::trace_event_data_t *TraceEvent =
      createTraceEvent(UserObj, "sampled_image_accessor", IId, CodeLoc,
                       xpti::trace_offload_accessor_event);
  xptiNotifySubscribers(GImageStreamID, NotificationTraceType, nullptr,
                        TraceEvent, IId, &AccessorConstr);
#endif
}

void XPTIRegistry::sampledImageHostAccessorNotification(
    const void *UserObj, const void *AccessorObj, const void *Type,
    uint32_t ElemSize, const detail::code_location &CodeLoc) {
  (void)UserObj;
  (void)AccessorObj;
  (void)CodeLoc;
  (void)Type;
  (void)ElemSize;
#ifdef XPTI_ENABLE_INSTRUMENTATION
  constexpr uint16_t NotificationTraceType = xpti::trace_offload_alloc_accessor;
  if (!xptiCheckTraceEnabled(GImageStreamID, NotificationTraceType))
    return;

  uint64_t IId;
  xpti::offload_image_accessor_data_t AccessorConstr{
      (uintptr_t)UserObj, (uintptr_t)AccessorObj, std::nullopt,
      std::nullopt,       (const char *)Type,     ElemSize};

  xpti::trace_event_data_t *TraceEvent =
      createTraceEvent(UserObj, "host_sampled_image_accessor", IId, CodeLoc,
                       xpti::trace_offload_accessor_event);
  xptiNotifySubscribers(GImageStreamID, NotificationTraceType, nullptr,
                        TraceEvent, IId, &AccessorConstr);
#endif
}

std::string queueDeviceToString(const queue_impl *const &Queue) {
  if (!Queue)
    return "HOST";
  auto Device = Queue->get_device();
  if (Device.is_cpu())
    return "CPU";
  else if (Device.is_gpu())
    return "GPU";
  else if (Device.is_accelerator())
    return "ACCELERATOR";
  else
    return "UNKNOWN";
}

} // namespace detail
} // namespace _V1
} // namespace sycl
