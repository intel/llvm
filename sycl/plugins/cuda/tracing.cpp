//===-------------- tracing.cpp - CUDA Host API Tracing --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include <xpti/xpti_data_types.h>
#include <xpti/xpti_trace_framework.h>
#endif

#include <cuda.h>
#include <cupti.h>

#include <exception>
#include <iostream>

constexpr auto CUDA_CALL_STREAM_NAME = "sycl.experimental.cuda.call";
constexpr auto CUDA_DEBUG_STREAM_NAME = "sycl.experimental.cuda.debug";

thread_local uint64_t CallCorrelationID = 0;
thread_local uint64_t DebugCorrelationID = 0;

#ifdef XPTI_ENABLE_INSTRUMENTATION
static xpti_td *GCallEvent = nullptr;
static xpti_td *GDebugEvent = nullptr;
#endif // XPTI_ENABLE_INSTRUMENTATION

constexpr auto GVerStr = "0.1";
constexpr int GMajVer = 0;
constexpr int GMinVer = 1;

#ifdef XPTI_ENABLE_INSTRUMENTATION
static void cuptiCallback(void *userdata, CUpti_CallbackDomain,
                          CUpti_CallbackId CBID, const void *CBData) {
  if (xptiTraceEnabled()) {
    const auto *CBInfo = static_cast<const CUpti_CallbackData *>(CBData);

    if (CBInfo->callbackSite == CUPTI_API_ENTER) {
      CallCorrelationID = xptiGetUniqueId();
      DebugCorrelationID = xptiGetUniqueId();
    }

    const char *FuncName = CBInfo->functionName;
    uint32_t FuncID = static_cast<uint32_t>(CBID);
    uint16_t TraceTypeArgs = CBInfo->callbackSite == CUPTI_API_ENTER
                                 ? xpti::trace_function_with_args_begin
                                 : xpti::trace_function_with_args_end;
    uint16_t TraceType = CBInfo->callbackSite == CUPTI_API_ENTER
                             ? xpti::trace_function_begin
                             : xpti::trace_function_end;

    uint8_t CallStreamID = xptiRegisterStream(CUDA_CALL_STREAM_NAME);
    uint8_t DebugStreamID = xptiRegisterStream(CUDA_DEBUG_STREAM_NAME);

    xptiNotifySubscribers(CallStreamID, TraceType, GCallEvent, nullptr,
                          CallCorrelationID, FuncName);

    xpti::function_with_args_t Payload{
        FuncID, FuncName, const_cast<void *>(CBInfo->functionParams),
        CBInfo->functionReturnValue, CBInfo->context};
    xptiNotifySubscribers(DebugStreamID, TraceTypeArgs, GDebugEvent, nullptr,
                          DebugCorrelationID, &Payload);
  }
}
#endif

void enableCUDATracing() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;

  xptiRegisterStream(CUDA_CALL_STREAM_NAME);
  xptiInitialize(CUDA_CALL_STREAM_NAME, GMajVer, GMinVer, GVerStr);
  xptiRegisterStream(CUDA_DEBUG_STREAM_NAME);
  xptiInitialize(CUDA_DEBUG_STREAM_NAME, GMajVer, GMinVer, GVerStr);

  uint64_t Dummy;
  xpti::payload_t CUDAPayload("CUDA Plugin Layer");
  GCallEvent =
      xptiMakeEvent("CUDA Plugin Layer", &CUDAPayload,
                    xpti::trace_algorithm_event, xpti_at::active, &Dummy);

  xpti::payload_t CUDADebugPayload("CUDA Plugin Debug Layer");
  GDebugEvent =
      xptiMakeEvent("CUDA Plugin Debug Layer", &CUDADebugPayload,
                    xpti::trace_algorithm_event, xpti_at::active, &Dummy);

  CUpti_SubscriberHandle Subscriber;
  cuptiSubscribe(&Subscriber, cuptiCallback, nullptr);
  cuptiEnableDomain(1, Subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  cuptiEnableCallback(0, Subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                      CUPTI_DRIVER_TRACE_CBID_cuGetErrorString);
  cuptiEnableCallback(0, Subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                      CUPTI_DRIVER_TRACE_CBID_cuGetErrorName);
#endif
}

void disableCUDATracing() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;

  xptiFinalize(CUDA_CALL_STREAM_NAME);
  xptiFinalize(CUDA_DEBUG_STREAM_NAME);
#endif // XPTI_ENABLE_INSTRUMENTATION
}
