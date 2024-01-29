//===-------------- tracing.cpp - CUDA Host API Tracing --------------------==//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef XPTI_ENABLE_INSTRUMENTATION
#include <xpti/xpti_data_types.h>
#include <xpti/xpti_trace_framework.h>
#endif

#include <cuda.h>
#ifdef XPTI_ENABLE_INSTRUMENTATION
#include <cupti.h>
#endif // XPTI_ENABLE_INSTRUMENTATION

#include "tracing.hpp"
#include "ur_lib_loader.hpp"
#include <exception>
#include <iostream>

#ifdef XPTI_ENABLE_INSTRUMENTATION
using tracing_event_t = xpti_td *;
using subscriber_handle_t = CUpti_SubscriberHandle;

using cuptiSubscribe_fn = CUPTIAPI
CUptiResult (*)(CUpti_SubscriberHandle *subscriber, CUpti_CallbackFunc callback,
                void *userdata);

using cuptiUnsubscribe_fn = CUPTIAPI
CUptiResult (*)(CUpti_SubscriberHandle subscriber);

using cuptiEnableDomain_fn = CUPTIAPI
CUptiResult (*)(uint32_t enable, CUpti_SubscriberHandle subscriber,
                CUpti_CallbackDomain domain);

using cuptiEnableCallback_fn = CUPTIAPI
CUptiResult (*)(uint32_t enable, CUpti_SubscriberHandle subscriber,
                CUpti_CallbackDomain domain, CUpti_CallbackId cbid);

#define LOAD_CUPTI_SYM(p, lib, x)                                              \
  p.x = (cupti##x##_fn)ur_loader::LibLoader::getFunctionPtr(lib.get(),         \
                                                            "cupti" #x);

#else
using tracing_event_t = void *;
using subscriber_handle_t = void *;
using cuptiSubscribe_fn = void *;
using cuptiUnsubscribe_fn = void *;
using cuptiEnableDomain_fn = void *;
using cuptiEnableCallback_fn = void *;
#endif // XPTI_ENABLE_INSTRUMENTATION

struct cupti_table_t_ {
  cuptiSubscribe_fn Subscribe = nullptr;
  cuptiUnsubscribe_fn Unsubscribe = nullptr;
  cuptiEnableDomain_fn EnableDomain = nullptr;
  cuptiEnableCallback_fn EnableCallback = nullptr;

  bool isInitialized() const;
};

struct cuda_tracing_context_t_ {
  tracing_event_t CallEvent = nullptr;
  tracing_event_t DebugEvent = nullptr;
  subscriber_handle_t Subscriber = nullptr;
  ur_loader::LibLoader::Lib Library;
  cupti_table_t_ Cupti;
};

#ifdef XPTI_ENABLE_INSTRUMENTATION
constexpr auto CUDA_CALL_STREAM_NAME = "sycl.experimental.cuda.call";
constexpr auto CUDA_DEBUG_STREAM_NAME = "sycl.experimental.cuda.debug";

thread_local uint64_t CallCorrelationID = 0;
thread_local uint64_t DebugCorrelationID = 0;

constexpr auto GVerStr = "0.1";
constexpr int GMajVer = 0;
constexpr int GMinVer = 1;

static void cuptiCallback(void *UserData, CUpti_CallbackDomain,
                          CUpti_CallbackId CBID, const void *CBData) {
  if (xptiTraceEnabled()) {
    const auto *CBInfo = static_cast<const CUpti_CallbackData *>(CBData);
    cuda_tracing_context_t_ *Ctx =
        static_cast<cuda_tracing_context_t_ *>(UserData);

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

    xptiNotifySubscribers(CallStreamID, TraceType, Ctx->CallEvent, nullptr,
                          CallCorrelationID, FuncName);

    xpti::function_with_args_t Payload{
        FuncID, FuncName, const_cast<void *>(CBInfo->functionParams),
        CBInfo->functionReturnValue, CBInfo->context};
    xptiNotifySubscribers(DebugStreamID, TraceTypeArgs, Ctx->DebugEvent,
                          nullptr, DebugCorrelationID, &Payload);
  }
}
#endif

cuda_tracing_context_t_ *createCUDATracingContext() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return nullptr;
  return new cuda_tracing_context_t_;
#else
  return nullptr;
#endif // XPTI_ENABLE_INSTRUMENTATION
}

void freeCUDATracingContext(cuda_tracing_context_t_ *Ctx) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  unloadCUDATracingLibrary(Ctx);
  delete Ctx;
#else
  (void)Ctx;
#endif // XPTI_ENABLE_INSTRUMENTATION
}

bool cupti_table_t_::isInitialized() const {
  return Subscribe && Unsubscribe && EnableDomain && EnableCallback;
}

bool loadCUDATracingLibrary(cuda_tracing_context_t_ *Ctx) {
#if defined(XPTI_ENABLE_INSTRUMENTATION) && defined(CUPTI_LIB_PATH)
  if (!Ctx)
    return false;
  if (Ctx->Library)
    return true;
  auto Lib{ur_loader::LibLoader::loadAdapterLibrary(CUPTI_LIB_PATH)};
  if (!Lib)
    return false;
  cupti_table_t_ Table;
  LOAD_CUPTI_SYM(Table, Lib, Subscribe)
  LOAD_CUPTI_SYM(Table, Lib, Unsubscribe)
  LOAD_CUPTI_SYM(Table, Lib, EnableDomain)
  LOAD_CUPTI_SYM(Table, Lib, EnableCallback)
  if (!Table.isInitialized()) {
    return false;
  }
  Ctx->Library = std::move(Lib);
  Ctx->Cupti = Table;
  return true;
#else
  (void)Ctx;
  return false;
#endif // XPTI_ENABLE_INSTRUMENTATION && CUPTI_LIB_PATH
}

void unloadCUDATracingLibrary(cuda_tracing_context_t_ *Ctx) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!Ctx)
    return;
  Ctx->Library.reset();
  Ctx->Cupti = cupti_table_t_();
#else
  (void)Ctx;
#endif // XPTI_ENABLE_INSTRUMENTATION
}

void enableCUDATracing(cuda_tracing_context_t_ *Ctx) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!Ctx || !xptiTraceEnabled())
    return;
  else if (!loadCUDATracingLibrary(Ctx))
    return;

  xptiRegisterStream(CUDA_CALL_STREAM_NAME);
  xptiInitialize(CUDA_CALL_STREAM_NAME, GMajVer, GMinVer, GVerStr);
  xptiRegisterStream(CUDA_DEBUG_STREAM_NAME);
  xptiInitialize(CUDA_DEBUG_STREAM_NAME, GMajVer, GMinVer, GVerStr);

  uint64_t Dummy;
  xpti::payload_t CUDAPayload("CUDA Plugin Layer");
  Ctx->CallEvent =
      xptiMakeEvent("CUDA Plugin Layer", &CUDAPayload,
                    xpti::trace_algorithm_event, xpti_at::active, &Dummy);

  xpti::payload_t CUDADebugPayload("CUDA Plugin Debug Layer");
  Ctx->DebugEvent =
      xptiMakeEvent("CUDA Plugin Debug Layer", &CUDADebugPayload,
                    xpti::trace_algorithm_event, xpti_at::active, &Dummy);

  Ctx->Cupti.Subscribe(&Ctx->Subscriber, cuptiCallback, Ctx);
  Ctx->Cupti.EnableDomain(1, Ctx->Subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  Ctx->Cupti.EnableCallback(0, Ctx->Subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuGetErrorString);
  Ctx->Cupti.EnableCallback(0, Ctx->Subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuGetErrorName);
#else
  (void)Ctx;
#endif
}

void disableCUDATracing(cuda_tracing_context_t_ *Ctx) {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!Ctx || !xptiTraceEnabled())
    return;

  if (Ctx->Subscriber && Ctx->Cupti.isInitialized()) {
    Ctx->Cupti.Unsubscribe(Ctx->Subscriber);
    Ctx->Subscriber = nullptr;
  }

  xptiFinalize(CUDA_CALL_STREAM_NAME);
  xptiFinalize(CUDA_DEBUG_STREAM_NAME);
#else
  (void)Ctx;
#endif // XPTI_ENABLE_INSTRUMENTATION
}
