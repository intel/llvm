//==----------- pi_trace.cpp.cpp -------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi_trace.cpp
/// A sample XPTI subscriber to demonstrate how to collect PI function call
/// arguments.

#include "xpti/xpti_trace_framework.h"

#include "pi_arguments_handler.hpp"

#include <detail/plugin_printers.hpp>

#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

static uint8_t GStreamID = 0;
std::mutex GIOMutex;

sycl::xpti_helpers::PiArgumentsHandler ArgHandler;

// The lone callback function we are going to use to demonstrate how to attach
// the collector to the running executable
XPTI_CALLBACK_API void tpCallback(uint16_t trace_type,
                                  xpti::trace_event_data_t *parent,
                                  xpti::trace_event_data_t *event,
                                  uint64_t instance, const void *user_data);

// Based on the documentation, every subscriber MUST implement the
// xptiTraceInit() and xptiTraceFinish() APIs for their subscriber collector to
// be loaded successfully.
XPTI_CALLBACK_API void xptiTraceInit(unsigned int /*major_version*/,
                                     unsigned int /*minor_version*/,
                                     const char * /*version_str*/,
                                     const char *stream_name) {
  if (std::string_view(stream_name) == "sycl.pi.debug") {
    GStreamID = xptiRegisterStream(stream_name);
    xptiRegisterCallback(GStreamID, xpti::trace_function_with_args_begin,
                         tpCallback);
    xptiRegisterCallback(GStreamID, xpti::trace_function_with_args_end,
                         tpCallback);

#define _PI_API(api)                                                           \
  ArgHandler.set##_##api(                                                      \
      [](const pi_plugin &, std::optional<pi_result>, auto &&...Args) {        \
        std::cout << "---> " << #api << "("                                    \
                  << "\n";                                                     \
        sycl::detail::pi::printArgs(Args...);                                  \
        std::cout << ") ---> ";                                                \
      });
#include <CL/sycl/detail/pi.def>
#undef _PI_API
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char * /*stream_name*/) {
  // NOP
}

XPTI_CALLBACK_API void tpCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t * /*Parent*/,
                                  xpti::trace_event_data_t * /*Event*/,
                                  uint64_t /*Instance*/, const void *UserData) {
  if (TraceType == xpti::trace_function_with_args_end) {
    // Lock while we print information
    std::lock_guard<std::mutex> Lock(GIOMutex);

    const auto *Data =
        static_cast<const xpti::function_with_args_t *>(UserData);
    const auto *Plugin = static_cast<pi_plugin *>(Data->user_data);

    ArgHandler.handle(Data->function_id, *Plugin, std::nullopt,
                      Data->args_data);
    std::cout << *static_cast<pi_result *>(Data->ret_data) << "\n";
  }
}
