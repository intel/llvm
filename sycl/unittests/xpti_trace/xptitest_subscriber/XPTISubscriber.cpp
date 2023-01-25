//==---------------------- XPTISubscriber.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xpti/xpti_trace_framework.h"

#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

std::deque<std::pair<uint16_t, std::string>> GReceivedNotifications;
XPTI_CALLBACK_API void testCallback(uint16_t TraceType,
                                    xpti::trace_event_data_t * /*Parent*/,
                                    xpti::trace_event_data_t *Event,
                                    uint64_t /*Instance*/,
                                    const void *UserData) {
  if (TraceType == xpti::trace_diagnostics) {
    std::string AggregatedData;
    if (Event && Event->reserved.payload) {
      auto Payload = Event->reserved.payload;
      const char Delimiter[] = ";";
      AggregatedData.append(Payload->name);
      AggregatedData.append(Delimiter);
      AggregatedData.append(Payload->source_file);
      AggregatedData.append(Delimiter);
      AggregatedData.append(std::to_string(Payload->line_no) + Delimiter +
                            std::to_string(Payload->column_no) + Delimiter);
    } else
      AggregatedData.append("code location unknown;");
    AggregatedData.append(static_cast<const char *>(UserData));
    GReceivedNotifications.push_back(std::make_pair(TraceType, AggregatedData));
  }
}

XPTI_CALLBACK_API void xptiTraceInit(unsigned int /*major_version*/,
                                     unsigned int /*minor_version*/,
                                     const char * /*version_str*/,
                                     const char *StreamName) {
  uint8_t StreamID = xptiRegisterStream("sycl.api");
  xptiRegisterCallback(StreamID, xpti::trace_diagnostics, testCallback);
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *StreamName) {}

XPTI_CALLBACK_API bool queryReceivedNotifications(uint16_t &TraceType,
                                                  std::string &Message) {
  if (GReceivedNotifications.empty())
    return false;
  auto &[traceType, message] = GReceivedNotifications.front();
  TraceType = traceType;
  Message = message;
  GReceivedNotifications.pop_front();
  return true;
}

XPTI_CALLBACK_API void resetReceivedNotifications() {
  GReceivedNotifications.clear();
}
