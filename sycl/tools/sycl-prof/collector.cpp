//==-------------- collector.cpp -------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "writer.hpp"
#include "xpti/xpti_data_types.h"

#include <cstdint>
#include <xpti/xpti_trace_framework.h>

#include <chrono>
#include <memory>
#include <stdio.h>
#include <thread>
#include <unistd.h>

namespace chrono = std::chrono;

Writer *GWriter = nullptr;

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
  size_t TS = chrono::time_point_cast<chrono::nanoseconds>(Now)
                  .time_since_epoch()
                  .count();

  return Measurements{TID, PID, TS};
}

XPTI_CALLBACK_API void piBeginEndCallback(uint16_t TraceType,
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
    GWriter = new JSONWriter(std::getenv("SYCL_PROF_OUT_FILE"));
    GWriter->init();
  }

  if (std::string_view(StreamName) == "sycl.pi") {
    uint8_t StreamID = xptiRegisterStream(StreamName);
    xptiRegisterCallback(StreamID, xpti::trace_function_begin,
                         piBeginEndCallback);
    xptiRegisterCallback(StreamID, xpti::trace_function_end,
                         piBeginEndCallback);
  } else if (std::string_view(StreamName) == "sycl") {
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
  }
}

XPTI_CALLBACK_API void xptiTraceFinish(const char *) { GWriter->finalize(); }

XPTI_CALLBACK_API void piBeginEndCallback(uint16_t TraceType,
                                          xpti::trace_event_data_t *,
                                          xpti::trace_event_data_t *,
                                          uint64_t /*Instance*/,
                                          const void *UserData) {
  auto [TID, PID, TS] = measure();
  if (TraceType == xpti::trace_function_begin) {
    GWriter->writeBegin(static_cast<const char *>(UserData), "Plugin", PID, TID,
                        TS);
  } else {
    GWriter->writeEnd(static_cast<const char *>(UserData), "Plugin", PID, TID,
                      TS);
  }
}

XPTI_CALLBACK_API void taskBeginEndCallback(uint16_t TraceType,
                                            xpti::trace_event_data_t *,
                                            xpti::trace_event_data_t *Event,
                                            uint64_t /*Instance*/,
                                            const void *) {
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
    GWriter->writeBegin(Name, "SYCL", PID, TID, TS);
  } else {
    GWriter->writeEnd(Name, "SYCL", PID, TID, TS);
  }
}

XPTI_CALLBACK_API void waitBeginEndCallback(uint16_t TraceType,
                                            xpti::trace_event_data_t *,
                                            xpti::trace_event_data_t *,
                                            uint64_t /*Instance*/,
                                            const void *UserData) {
  auto [TID, PID, TS] = measure();
  if (TraceType == xpti::trace_wait_begin ||
      TraceType == xpti::trace_barrier_begin) {
    GWriter->writeBegin(static_cast<const char *>(UserData), "SYCL", PID, TID,
                        TS);
  } else {
    GWriter->writeEnd(static_cast<const char *>(UserData), "SYCL", PID, TID,
                      TS);
  }
}
