//==-------------- collector.cpp -------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xpti/xpti_data_types.h"
#include "xpti/xpti_trace_framework.h"
#include "xpti/xpti_trace_writer.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <thread>
#include <unistd.h>

namespace chrono = std::chrono;

xpti::trace::TraceWriter *GWriter = nullptr;

struct Measurements {
  size_t TID;
  size_t PID;
  size_t TimeStamp;
};

unsigned long process_id() { return static_cast<unsigned long>(getpid()); }

static Measurements measure() {
  size_t TID = std::hash<std::thread::id>{}(std::this_thread::get_id());
  size_t PID = process_id();
  auto Now = chrono::high_resolution_clock::now();
  size_t TS = chrono::time_point_cast<chrono::microseconds>(Now)
                  .time_since_epoch()
                  .count();

  return Measurements{TID, PID, TS};
}

XPTI_CALLBACK_API void apiBeginEndCallback(uint16_t TraceType,
                                           xpti::trace_event_data_t *,
                                           xpti::trace_event_data_t *,
                                           uint64_t /*Instance*/,
                                           const void *UserData);
XPTI_CALLBACK_API void taskBeginEndCallback(uint16_t TraceType,
                                            xpti::trace_event_data_t *,
                                            xpti::trace_event_data_t *,
                                            uint64_t /*Instance*/,
                                            const void *UserData);
XPTI_CALLBACK_API void waitBeginEndCallback(uint16_t TraceType,
                                            xpti::trace_event_data_t *,
                                            xpti::trace_event_data_t *,
                                            uint64_t /*Instance*/,
                                            const void *UserData);

XPTI_CALLBACK_API void xptiTraceInit(unsigned int /*major_version*/,
                                     unsigned int /*minor_version*/,
                                     const char * /*version_str*/,
                                     const char *StreamName) {
  if (GWriter == nullptr) {
    const char *ProfOutFile = std::getenv("SYCL_PROF_OUT_FILE");
    if (!ProfOutFile)
      throw std::runtime_error(
          "SYCL_PROF_OUT_FILE environment variable is not specified");
    GWriter = new xpti::trace::JSONWriter(ProfOutFile);
  }

  std::string_view NameView{StreamName};
  if (NameView == "sycl.pi") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_begin,
                         apiBeginEndCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_end,
                         apiBeginEndCallback);
  } else if (NameView == "sycl") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_task_begin,
                         taskBeginEndCallback);
    xptiRegisterCallback(StreamID, xpti::trace_task_end, taskBeginEndCallback);
    xptiRegisterCallback(StreamID, xpti::trace_wait_begin,
                         waitBeginEndCallback);
    xptiRegisterCallback(StreamID, xpti::trace_wait_end, waitBeginEndCallback);
    xptiRegisterCallback(StreamID, xpti::trace_barrier_begin,
                         waitBeginEndCallback);
    xptiRegisterCallback(StreamID, xpti::trace_barrier_end,
                         waitBeginEndCallback);
  } else if (NameView == "sycl.experimental.level_zero.call") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_begin,
                         apiBeginEndCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_end,
                         apiBeginEndCallback);
  } else if (NameView == "sycl.experimental.cuda.call") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_begin,
                         apiBeginEndCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_end,
                         apiBeginEndCallback);
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *) { 
  if (GWriter != nullptr) {
    delete GWriter;
    GWriter = nullptr;
  }
}

XPTI_CALLBACK_API void apiBeginEndCallback(uint16_t TraceType,
                                           xpti::trace_event_data_t *,
                                           xpti::trace_event_data_t *Event,
                                           uint64_t Instance,
                                           const void *UserData) {
  if (!GWriter)
    return;

  auto [TID, PID, TS] = measure();
  if (TraceType == xpti::trace_function_begin) {
    GWriter->addBeginEvent(static_cast<const char *>(UserData), {"API"},
                           Instance, Event, PID, TID, TS);
  } else {
    GWriter->addEndEvent(static_cast<const char *>(UserData), {"API"}, Instance,
                         Event, PID, TID, TS);
  }
}

XPTI_CALLBACK_API void taskBeginEndCallback(uint16_t TraceType,
                                            xpti::trace_event_data_t *,
                                            xpti::trace_event_data_t *Event,
                                            uint64_t Instance, const void *) {
  if (!GWriter)
    return;

  std::string_view Name = "unknown";

  xpti::metadata_t *Metadata = xptiQueryMetadata(Event);
  for (auto &Item : *Metadata) {
    std::string_view Key{xptiLookupString(Item.first)};
    if (Key == "kernel_name" || Key == "memory_object") {
      Name = xptiLookupString(Item.second);
    }
  }

  auto [TID, PID, TS] = measure();
  if (TraceType == xpti::trace_task_begin) {
    GWriter->addBeginEvent(Name.data(), {"SYCL", "Task"}, Instance, Event, PID,
                           TID, TS);
  } else {
    GWriter->addEndEvent(Name.data(), {"SYCL", "Task"}, Instance, Event, PID,
                         TID, TS);
  }
}

XPTI_CALLBACK_API void waitBeginEndCallback(uint16_t TraceType,
                                            xpti::trace_event_data_t *,
                                            xpti::trace_event_data_t *Event,
                                            uint64_t Instance,
                                            const void *UserData) {
  if (!GWriter)
    return;

  auto [TID, PID, TS] = measure();
  if (TraceType == xpti::trace_wait_begin ||
      TraceType == xpti::trace_barrier_begin) {
    GWriter->addBeginEvent(static_cast<const char *>(UserData),
                           {"SYCL", "Wait"}, Instance, Event, PID, TID, TS);
  } else {
    GWriter->addEndEvent(static_cast<const char *>(UserData), {"SYCL", "Wait"},
                         Instance, Event, PID, TID, TS);
  }
}
