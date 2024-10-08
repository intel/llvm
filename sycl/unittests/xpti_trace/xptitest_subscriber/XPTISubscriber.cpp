//==---------------------- XPTISubscriber.cpp ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xpti/xpti_trace_framework.hpp"

#include <deque>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>

std::deque<std::pair<uint16_t, std::string>> GReceivedNotifications;
std::set<uint16_t> GAnalyzedTraceTypes;

XPTI_CALLBACK_API void addAnalyzedTraceType(uint16_t TraceType) {
  GAnalyzedTraceTypes.insert(TraceType);
}

XPTI_CALLBACK_API void clearAnalyzedTraceTypes() {
  GAnalyzedTraceTypes.clear();
}

XPTI_CALLBACK_API void testCallback(uint16_t TraceType,
                                    xpti::trace_event_data_t * /*Parent*/,
                                    xpti::trace_event_data_t *Event,
                                    uint64_t /*Instance*/,
                                    const void *UserData) {
  if (GAnalyzedTraceTypes.find(TraceType) == GAnalyzedTraceTypes.end())
    return;

  // Since "queue_id" is no longer a metadata item, we have to retrieve it from
  // TLS using new XPTI API
  char *Key = 0;
  uint64_t Value;
  bool HaveKeyValue =
      (xptiGetStashedTuple(&Key, Value) == xpti::result_t::XPTI_RESULT_SUCCESS);

  if (TraceType == xpti::trace_diagnostics) {
    std::string AggregatedData;
    if (Event && Event->reserved.payload && Event->reserved.payload->name &&
        Event->reserved.payload->source_file) {
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
  } else if (TraceType == xpti::trace_node_create) {
    std::string UData(static_cast<const char *>(UserData));
    if (UData.find("command_group_node") != std::string::npos) {
      auto Payload = xptiQueryPayload(Event);
      xpti::metadata_t *Metadata = xptiQueryMetadata(Event);
      for (const auto &Item : *Metadata) {
        std::string_view Key{xptiLookupString(Item.first)};
        if (Key == "kernel_name") {
          GReceivedNotifications.push_back(
              std::make_pair(TraceType, UData + std::string(Payload->name)));
        }
      }
    } else if (UData.find("memory_transfer_node") != std::string::npos) {
      GReceivedNotifications.push_back(std::make_pair(TraceType, UData));
    }
  } else if (TraceType == xpti::trace_queue_create) {
    if (Event) {
      std::string Message;
      xpti::metadata_t *Metadata = xptiQueryMetadata(Event);
      for (const auto &Item : *Metadata) {
        std::string_view Key{xptiLookupString(Item.first)};
        if (Key == "queue_id") {
          Message.append(
              std::string("create:") + Key.data() + std::string(":") +
              std::to_string(
                  xpti::getMetadata<unsigned long long>(Item).second));
          Message.append(";");
        } else if (Key == "queue_handle") {
          Message.append(
              Key.data() + std::string(":") +
              std::to_string(xpti::getMetadata<size_t>(Item).second));
          Message.append(";");
        }
      }
      GReceivedNotifications.push_back(std::make_pair(TraceType, Message));
    }
  } else if (TraceType == xpti::trace_queue_destroy) {
    if (Event) {
      std::string Message;
      xpti::metadata_t *Metadata = xptiQueryMetadata(Event);
      for (const auto &Item : *Metadata) {
        std::string_view Key{xptiLookupString(Item.first)};
        if (Key == "queue_id") {
          Message.append(
              std::string("destroy:") + Key.data() + std::string(":") +
              std::to_string(
                  xpti::getMetadata<unsigned long long>(Item).second));
          Message.append(";");
        } else if (Key == "queue_handle") {
          Message.append(
              Key.data() + std::string(":") +
              std::to_string(xpti::getMetadata<size_t>(Item).second));
          Message.append(";");
        }
      }
      GReceivedNotifications.push_back(std::make_pair(TraceType, Message));
    }
  } else if (TraceType == xpti::trace_task_begin) {
    if (Event) {
      std::string Message;
      // Since we have changed we send the "queue_id" information, we no longer
      // have to check the metadata for the instance ID
      if (HaveKeyValue) {
        Message.append(std::string("task_begin:") + Key + std::string(":") +
                       std::to_string(Value));
      }
      GReceivedNotifications.push_back(std::make_pair(TraceType, Message));
    }
  } else if (TraceType == xpti::trace_task_end) {
    if (Event) {
      std::string Message;
      // Since we have changed we send the "queue_id" information, we no longer
      // have to check the metadata for the instance ID
      if (HaveKeyValue) {
        Message.append(std::string("task_end:") + Key + std::string(":") +
                       std::to_string(Value));
      }
      GReceivedNotifications.push_back(std::make_pair(TraceType, Message));
    }
  }
}

XPTI_CALLBACK_API void xptiTraceInit(unsigned int /*major_version*/,
                                     unsigned int /*minor_version*/,
                                     const char * /*version_str*/,
                                     const char * /*StreamName*/) {
  uint8_t StreamID = xptiRegisterStream("sycl");
  xptiRegisterCallback(StreamID, xpti::trace_diagnostics, testCallback);
  xptiRegisterCallback(StreamID, xpti::trace_node_create, testCallback);
  xptiRegisterCallback(StreamID, xpti::trace_task_begin, testCallback);
  xptiRegisterCallback(StreamID, xpti::trace_task_end, testCallback);
  xptiRegisterCallback(StreamID, xpti::trace_queue_create, testCallback);
  xptiRegisterCallback(StreamID, xpti::trace_queue_destroy, testCallback);
  xptiRegisterCallback(StreamID, xpti::trace_task_begin, testCallback);
  xptiRegisterCallback(StreamID, xpti::trace_task_end, testCallback);
}

XPTI_CALLBACK_API void xptiTraceFinish(const char * /*StreamName*/) {}

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
