//==---------------------- collector.cpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xpti/xpti_trace_framework.h"

#include <sycl/detail/spinlock.hpp>

#include <iostream>

sycl::detail::SpinLock GlobalLock;

bool HasZEPrinter = false;
bool HasCUPrinter = false;
bool HasPIPrinter = false;
bool HasSYCLPrinter = false;

void zePrintersInit();
void zePrintersFinish();
#ifdef USE_PI_CUDA
void cuPrintersInit();
void cuPrintersFinish();
#endif
void piPrintersInit();
void piPrintersFinish();
void syclPrintersInit();
void syclPrintersFinish();

XPTI_CALLBACK_API void piCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t *Parent,
                                  xpti::trace_event_data_t *Event,
                                  uint64_t Instance, const void *UserData);
XPTI_CALLBACK_API void zeCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t *Parent,
                                  xpti::trace_event_data_t *Event,
                                  uint64_t Instance, const void *UserData);
#ifdef USE_PI_CUDA
XPTI_CALLBACK_API void cuCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t *Parent,
                                  xpti::trace_event_data_t *Event,
                                  uint64_t Instance, const void *UserData);
#endif
XPTI_CALLBACK_API void syclCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t *Parent,
                                  xpti::trace_event_data_t *Event,
                                  uint64_t Instance, const void *UserData);
XPTI_CALLBACK_API void xptiTraceInit(unsigned int /*major_version*/,
                                     unsigned int /*minor_version*/,
                                     const char * /*version_str*/,
                                     const char *StreamName) {
  if (std::string_view(StreamName) == "sycl.pi.debug" &&
      std::getenv("SYCL_TRACE_PI_ENABLE")) {
    piPrintersInit();
    uint16_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_begin,
                         piCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_end,
                         piCallback);
#ifdef SYCL_HAS_LEVEL_ZERO
  } else if (std::string_view(StreamName) ==
                 "sycl.experimental.level_zero.debug" &&
             std::getenv("SYCL_TRACE_ZE_ENABLE")) {
    zePrintersInit();
    uint16_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_begin,
                         zeCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_end,
                         zeCallback);
#endif
#ifdef USE_PI_CUDA
  } else if (std::string_view(StreamName) == "sycl.experimental.cuda.debug" &&
             std::getenv("SYCL_TRACE_CU_ENABLE")) {
    cuPrintersInit();
    uint16_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_begin,
                         cuCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_end,
                         cuCallback);
#endif
  }
  if (std::string_view(StreamName) == "sycl.api" &&
      std::getenv("SYCL_TRACE_API_ENABLE")) {
    syclPrintersInit();
    uint16_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_diagnostics, syclCallback);
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *StreamName) {
  if (std::string_view(StreamName) == "sycl.pi.debug" &&
      std::getenv("SYCL_TRACE_PI_ENABLE"))
    piPrintersFinish();
#ifdef SYCL_HAS_LEVEL_ZERO
  else if (std::string_view(StreamName) ==
               "sycl.experimental.level_zero.debug" &&
           std::getenv("SYCL_TRACE_ZE_ENABLE"))
    zePrintersFinish();
#endif
#ifdef USE_PI_CUDA
  else if (std::string_view(StreamName) == "sycl.experimental.cuda.debug" &&
           std::getenv("SYCL_TRACE_CU_ENABLE"))
    cuPrintersFinish();
#endif
  if (std::string_view(StreamName) == "sycl.api" &&
      std::getenv("SYCL_TRACE_API_ENABLE"))
    syclPrintersFinish();
}
