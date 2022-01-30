//===-------------- tracing.cpp - L0 Host API Tracing ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xpti/xpti_data_types.h"
#include <level_zero/layers/zel_tracing_api.h>
#include <level_zero/ze_api.h>
#include <xpti/xpti_trace_framework.h>

#include <iostream>

constexpr auto L0_CALL_STREAM_NAME = "sycl.level_zero.call";
constexpr auto L0_DEBUG_STREAM_NAME = "sycl.level_zero.debug";

thread_local uint64_t CallCorrelationID = 0;
thread_local uint64_t DebugCorrelationID = 0;

#ifdef XPTI_ENABLE_INSTRUMENTATION
xpti_td *GCallEvent = nullptr;
xpti_td *GDebugEvent = nullptr;
#endif // XPTI_ENABLE_INSTRUMENTATION

void enableL0Tracing() {
#ifdef XPTI_ENABLE_INSTRUMENTATION
  zel_tracer_desc_t TracerDesc = {ZEL_STRUCTURE_TYPE_TRACER_EXP_DESC, nullptr,
                                  nullptr};
  zel_tracer_handle_t Tracer = nullptr;

  ze_result_t Status = zelTracerCreate(&TracerDesc, &Tracer);

  if (Status != ZE_RESULT_SUCCESS || Tracer == nullptr) {
    std::cerr << "[WARNING] Failed to create L0 tracer\n";
    return;
  }

  zel_core_callbacks_t Prologue = {};
  zel_core_callbacks_t Epilogue = {};

#define _ZE_API(call, cb, domain, params_type)                                 \
  Prologue.domain.cb = [](params_type *Params, ze_result_t, void *, void **) { \
    if (xptiTraceEnabled()) {                                                  \
      uint8_t CallStreamID = xptiRegisterStream(L0_CALL_STREAM_NAME);          \
      uint8_t DebugStreamID = xptiRegisterStream(L0_DEBUG_STREAM_NAME);        \
      CallCorrelationID = xptiGetUniqueId();                                   \
      DebugCorrelationID = xptiGetUniqueId();                                  \
      const char *FuncName = #call;                                            \
      xptiNotifySubscribers(                                                   \
          CallStreamID, (uint16_t)xpti::trace_point_type_t::function_begin,    \
          GCallEvent, nullptr, CallCorrelationID, FuncName);                   \
      xpti::function_with_args_t Payload{0, FuncName, Params, nullptr,         \
                                         nullptr};                             \
      xptiNotifySubscribers(                                                   \
          DebugStreamID,                                                       \
          (uint16_t)xpti::trace_point_type_t::function_with_args_begin,        \
          GDebugEvent, nullptr, DebugCorrelationID, &Payload);                 \
    }                                                                          \
  };                                                                           \
  Epilogue.domain.cb = [](params_type *Params, ze_result_t Result, void *,     \
                          void **) {                                           \
    if (xptiTraceEnabled()) {                                                  \
      uint8_t CallStreamID = xptiRegisterStream(L0_CALL_STREAM_NAME);          \
      uint8_t DebugStreamID = xptiRegisterStream(L0_DEBUG_STREAM_NAME);        \
      const char *FuncName = #call;                                            \
      xptiNotifySubscribers(CallStreamID,                                      \
                            (uint16_t)xpti::trace_point_type_t::function_end,  \
                            GCallEvent, nullptr, CallCorrelationID, FuncName); \
      xpti::function_with_args_t Payload{0, FuncName, Params, &Result,         \
                                         nullptr};                             \
      xptiNotifySubscribers(                                                   \
          DebugStreamID,                                                       \
          (uint16_t)xpti::trace_point_type_t::function_with_args_end,          \
          GDebugEvent, nullptr, DebugCorrelationID, &Payload);                 \
    }                                                                          \
  };

#include "ze_api.def"

#undef _ZE_API

  Status = zelTracerSetEnabled(Tracer, true);
  if (Status != ZE_RESULT_SUCCESS) {
    std::cerr << "Failed to enable L0 tracing\n";
    std::terminate();
  }
#endif
}
