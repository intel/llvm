//===-------------- tracing.cpp - Level-Zero Host API Tracing --------------==//
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

#include <exception>
#include <layers/zel_tracing_api.h>
#include <ze_api.h>

#include <sycl/detail/iostream_proxy.hpp>

constexpr auto ZE_CALL_STREAM_NAME = "sycl.experimental.level_zero.call";
constexpr auto ZE_DEBUG_STREAM_NAME = "sycl.experimental.level_zero.debug";

thread_local uint64_t CallCorrelationID = 0;
thread_local uint64_t DebugCorrelationID = 0;

constexpr auto GVerStr = "0.1";
constexpr int GMajVer = 0;
constexpr int GMinVer = 1;

#ifdef XPTI_ENABLE_INSTRUMENTATION
static xpti_td *GCallEvent = nullptr;
static xpti_td *GDebugEvent = nullptr;
static uint8_t GCallStreamID = 0;
static uint8_t GDebugStreamID = 0;
#endif // XPTI_ENABLE_INSTRUMENTATION

enum class ZEApiKind {
#define _ZE_API(call, domain, cb, params_type) call,
#include "ze_api.def"
#undef _ZE_API
};

void enableZeTracing() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;

  // Initialize the required streams and stream ID for use
  GCallStreamID = xptiRegisterStream(ZE_CALL_STREAM_NAME);
  xptiInitialize(ZE_CALL_STREAM_NAME, GMajVer, GMinVer, GVerStr);
  GDebugStreamID = xptiRegisterStream(ZE_DEBUG_STREAM_NAME);
  xptiInitialize(ZE_DEBUG_STREAM_NAME, GMajVer, GMinVer, GVerStr);

  uint64_t Dummy;
  xpti::payload_t ZePayload("Level Zero Plugin Layer");
  GCallEvent =
      xptiMakeEvent("Level Zero Plugin Layer", &ZePayload,
                    xpti::trace_algorithm_event, xpti_at::active, &Dummy);

  xpti::payload_t ZeDebugPayload("Level Zero Plugin Debug Layer");
  GDebugEvent =
      xptiMakeEvent("Level Zero Plugin Debug Layer", &ZeDebugPayload,
                    xpti::trace_algorithm_event, xpti_at::active, &Dummy);

  ze_result_t Status = zeInit(0);
  if (Status != ZE_RESULT_SUCCESS) {
    // Most likey there are no Level Zero devices.
    return;
  }

  int Foo = 0;
  zel_tracer_desc_t TracerDesc = {ZEL_STRUCTURE_TYPE_TRACER_EXP_DESC, nullptr,
                                  &Foo};
  zel_tracer_handle_t Tracer = nullptr;

  Status = zelTracerCreate(&TracerDesc, &Tracer);

  if (Status != ZE_RESULT_SUCCESS || Tracer == nullptr) {
    std::cerr << "[WARNING] Failed to create Level Zero tracer: " << Status
              << "\n";
    return;
  }

  zel_core_callbacks_t Prologue = {};
  zel_core_callbacks_t Epilogue = {};

#define _ZE_API(call, domain, cb, params_type)                                 \
  Prologue.domain.cb = [](params_type *Params, ze_result_t, void *, void **) { \
    if (xptiTraceEnabled()) {                                                  \
      const char *FuncName = #call;                                            \
      if (xptiCheckTraceEnabled(                                               \
              GCallStreamID,                                                   \
              (uint16_t)xpti::trace_point_type_t::function_begin)) {           \
        CallCorrelationID = xptiGetUniqueId();                                 \
        xptiNotifySubscribers(                                                 \
            GCallStreamID, (uint16_t)xpti::trace_point_type_t::function_begin, \
            GCallEvent, nullptr, CallCorrelationID, FuncName);                 \
      }                                                                        \
      if (xptiCheckTraceEnabled(                                               \
              GDebugStreamID,                                                  \
              (uint16_t)xpti::trace_point_type_t::function_with_args_begin)) { \
        DebugCorrelationID = xptiGetUniqueId();                                \
        uint32_t FuncID = static_cast<uint32_t>(ZEApiKind::call);              \
        xpti::function_with_args_t Payload{FuncID, FuncName, Params, nullptr,  \
                                           nullptr};                           \
        xptiNotifySubscribers(                                                 \
            GDebugStreamID,                                                    \
            (uint16_t)xpti::trace_point_type_t::function_with_args_begin,      \
            GDebugEvent, nullptr, DebugCorrelationID, &Payload);               \
      }                                                                        \
    }                                                                          \
  };                                                                           \
  Epilogue.domain.cb = [](params_type *Params, ze_result_t Result, void *,     \
                          void **) {                                           \
    if (xptiTraceEnabled()) {                                                  \
      const char *FuncName = #call;                                            \
      if (xptiCheckTraceEnabled(                                               \
              GCallStreamID,                                                   \
              (uint16_t)xpti::trace_point_type_t::function_end)) {             \
        xptiNotifySubscribers(                                                 \
            GCallStreamID, (uint16_t)xpti::trace_point_type_t::function_end,   \
            GCallEvent, nullptr, CallCorrelationID, FuncName);                 \
      }                                                                        \
      if (xptiCheckTraceEnabled(                                               \
              GDebugStreamID,                                                  \
              (uint16_t)xpti::trace_point_type_t::function_with_args_end)) {   \
        uint32_t FuncID = static_cast<uint32_t>(ZEApiKind::call);              \
        xpti::function_with_args_t Payload{FuncID, FuncName, Params, &Result,  \
                                           nullptr};                           \
        xptiNotifySubscribers(                                                 \
            GDebugStreamID,                                                    \
            (uint16_t)xpti::trace_point_type_t::function_with_args_end,        \
            GDebugEvent, nullptr, DebugCorrelationID, &Payload);               \
      }                                                                        \
    }                                                                          \
  };

#include "ze_api.def"

#undef _ZE_API

  Status = zelTracerSetPrologues(Tracer, &Prologue);
  if (Status != ZE_RESULT_SUCCESS) {
    std::cerr << "Failed to enable Level Zero tracing\n";
    std::terminate();
  }
  Status = zelTracerSetEpilogues(Tracer, &Epilogue);
  if (Status != ZE_RESULT_SUCCESS) {
    std::cerr << "Failed to enable Level Zero tracing\n";
    std::terminate();
  }

  Status = zelTracerSetEnabled(Tracer, true);
  if (Status != ZE_RESULT_SUCCESS) {
    std::cerr << "Failed to enable Level Zero tracing\n";
    std::terminate();
  }
#endif // XPTI_ENABLE_INSTRUMENTATION
}

void disableZeTracing() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  if (!xptiTraceEnabled())
    return;

  xptiFinalize(ZE_CALL_STREAM_NAME);
  xptiFinalize(ZE_DEBUG_STREAM_NAME);
#endif // XPTI_ENABLE_INSTRUMENTATION
}
