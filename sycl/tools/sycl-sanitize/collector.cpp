//==-------------- collector.cpp -------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file collector.cpp
/// The SYCL sanitizer collector intercepts PI calls to find memory leaks in
/// usages of USM pointers.

#include "xpti/xpti_trace_framework.h"

#include "pi_arguments_handler.hpp"
#include "usm_analyzer.hpp"

#include <detail/plugin_printers.hpp>

#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

std::mutex IOMutex;

XPTI_CALLBACK_API void tpCallback(uint16_t trace_type,
                                  xpti::trace_event_data_t *parent,
                                  xpti::trace_event_data_t *event,
                                  uint64_t instance, const void *user_data);

XPTI_CALLBACK_API void xptiTraceInit(unsigned int /*major_version*/,
                                     unsigned int /*minor_version*/,
                                     const char * /*version_str*/,
                                     const char *StreamName) {
  if (std::string_view(StreamName) == "sycl.pi.debug") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_begin,
                         tpCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_with_args_end,
                         tpCallback);
    auto &GS = USMAnalyzer::getInstance();
    GS.changeTerminationOnErrorState(true);
    GS.printToErrorStream();
    GS.setupUSMHandlers();
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *StreamName) {
  if (std::string_view(StreamName) == "sycl.pi.debug") {
    bool hadLeak = false;
    auto &GS = USMAnalyzer::getInstance();
    if (GS.ActivePointers.size() > 0) {
      hadLeak = true;
      std::cerr << "Found " << GS.ActivePointers.size()
                << " leaked memory allocations\n";
      for (const auto &Ptr : GS.ActivePointers) {
        std::cerr << "Leaked pointer: " << std::hex << Ptr.first << "\n";
        std::cerr << "  Location: "
                  << "function " << Ptr.second.Location.Function << " at "
                  << Ptr.second.Location.Source << ":" << std::dec
                  << Ptr.second.Location.Line << "\n";
      }
    }
    if (hadLeak)
      exit(-1);
  }
}

XPTI_CALLBACK_API void tpCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t *,
                                  xpti::trace_event_data_t *ObjectEvent,
                                  uint64_t /*Instance*/, const void *UserData) {
  auto &GS = USMAnalyzer::getInstance();
  GS.fillLastTracepointData(ObjectEvent);

  // Lock while we capture information
  std::lock_guard<std::mutex> Lock(IOMutex);

  const auto *Data = static_cast<const xpti::function_with_args_t *>(UserData);
  const auto *Plugin = static_cast<pi_plugin *>(Data->user_data);
  if (TraceType == xpti::trace_function_with_args_begin) {
    GS.ArgHandlerPreCall.handle(Data->function_id, *Plugin, std::nullopt,
                                Data->args_data);
  } else if (TraceType == xpti::trace_function_with_args_end) {
    const pi_result Result = *static_cast<pi_result *>(Data->ret_data);
    GS.ArgHandlerPostCall.handle(Data->function_id, *Plugin, Result,
                                 Data->args_data);
  }
}
