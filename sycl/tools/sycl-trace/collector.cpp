//==---------------------- collector.cpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xpti/xpti_trace_framework.h"

#include <sycl/detail/spinlock.hpp>

sycl::detail::SpinLock GlobalLock;

bool HasZEPrinter = false;
bool HasPIPrinter = false;

void zePrintersInit();
void zePrintersFinish();
void piPrintersInit();
void piPrintersFinish();

XPTI_CALLBACK_API void piCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t *Parent,
                                  xpti::trace_event_data_t *Event,
                                  uint64_t Instance, const void *UserData);
XPTI_CALLBACK_API void zeCallback(uint16_t TraceType,
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
  } else if (std::string_view(StreamName) ==
                 "sycl.experimental.level_zero.debug" &&
             std::getenv("SYCL_TRACE_ZE_ENABLE")) {
    zePrintersInit();
    uint16_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_begin,
                         zeCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_end,
                         zeCallback);
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *StreamName) {
  if (std::string_view(StreamName) == "sycl.pi.debug" &&
      std::getenv("SYCL_TRACE_PI_ENABLE"))
    piPrintersFinish();
  else if (std::string_view(StreamName) ==
               "sycl.experimental.level_zero.debug" &&
           std::getenv("SYCL_TRACE_ZE_ENABLE"))
    zePrintersFinish();
}
