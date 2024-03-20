//==----------- sycl_trace_collector.cpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file sycl_trace_collector.cpp
/// Routines to collect and print SYCL API calls.

#include "xpti/xpti_trace_framework.hpp"

#include <sycl/detail/spinlock.hpp>

#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

extern sycl::detail::SpinLock GlobalLock;

bool PrintSyclVerbose = false;

void TraceDiagnosticsMessage(xpti::trace_event_data_t * /*Parent*/,
                             xpti::trace_event_data_t *CurrentObject,
                             const char *Message) {
  if (!Message)
    return;

  std::cout << "[SYCL] Runtime reports: " << std::endl;
  std::cout << "what:  " << Message << std::endl;
  if (!CurrentObject)
    return;
  std::cout << "where: ";
  if (auto Payload = CurrentObject->reserved.payload) {
    bool HasData = false;
    if (Payload->flags & (uint64_t)xpti::payload_flag_t::SourceFileAvailable) {
      HasData = true;
      std::cout << Payload->source_file << ":" << Payload->line_no << "\t";
    }
    if (Payload->flags & (uint64_t)xpti::payload_flag_t::NameAvailable) {
      HasData = true;
      std::cout << Payload->name;
    }
    if (!HasData)
      std::cout << "No code location data is available.";
    std::cout << std::endl;
  }
}

void TraceTaskExecutionSignals(xpti::trace_event_data_t * /*Parent*/,
                               xpti::trace_event_data_t *Event,
                               [[maybe_unused]] const void *UserData,
                               uint64_t InstanceID, bool IsBegin) {
  if (!Event)
    return;

  char *Key = 0;
  uint64_t Value;
  bool HaveKeyValue =
      (xptiGetStashedTuple(&Key, Value) == xpti::result_t::XPTI_RESULT_SUCCESS);

  std::cout << "[SYCL] Task " << (IsBegin ? "begin" : "end  ")
            << " (event=" << Event << ",instanceID=" << InstanceID << ")"
            << std::endl;

  // TODO: some metadata could be added at the "end" point (e.g. memory
  // allocation and result ptr). To consider how to distinguish new data for the
  // same event appeared between begin-end points.
  if (!IsBegin || !PrintSyclVerbose)
    return;

  if (HaveKeyValue) {
    std::cout << "\t  " << Key << " : " << Value << std::endl;
  }

  xpti::metadata_t *Metadata = xptiQueryMetadata(Event);
  for (auto &Item : *Metadata) {
    std::cout << "\t  " << xptiLookupString(Item.first) << " : "
              << xpti::readMetadata(Item) << std::endl;
  }
}

void TraceQueueLifetimeSignals(xpti::trace_event_data_t * /*Parent*/,
                               xpti::trace_event_data_t *Event,
                               [[maybe_unused]] const void *UserData,
                               bool IsCreation) {
  if (!Event)
    return;

  std::cout << "[SYCL] Queue " << (IsCreation ? "create" : "destroy") << ": "
            << std::endl;
  xpti::metadata_t *Metadata = xptiQueryMetadata(Event);
  for (auto &Item : *Metadata) {
    std::string_view Key{xptiLookupString(Item.first)};
    if (IsCreation)
      std::cout << "\t  " << Key << " : " << xpti::readMetadata(Item)
                << std::endl;
    else if (Key == "queue_id")
      std::cout << "\t" << Key << " : "
                << xpti::getMetadata<unsigned long long>(Item).second
                << std::endl;
  }
}

XPTI_CALLBACK_API void syclCallback(uint16_t TraceType,
                                    xpti::trace_event_data_t *Parent,
                                    xpti::trace_event_data_t *Event,
                                    uint64_t InstanceID, const void *UserData) {
  std::lock_guard<sycl::detail::SpinLock> Lock{GlobalLock};
  switch (TraceType) {
  case xpti::trace_diagnostics:
    TraceDiagnosticsMessage(Parent, Event, static_cast<const char *>(UserData));
    break;
  case xpti::trace_queue_create:
    TraceQueueLifetimeSignals(Parent, Event, UserData, true);
    break;
  case xpti::trace_queue_destroy:
    TraceQueueLifetimeSignals(Parent, Event, UserData, false);
    break;
  case xpti::trace_task_begin:
    TraceTaskExecutionSignals(Parent, Event, UserData, InstanceID, true);
    break;
  case xpti::trace_task_end:
    TraceTaskExecutionSignals(Parent, Event, UserData, InstanceID, false);
    break;
  default: {
    if (PrintSyclVerbose)
      std::cout << "Trace type is unexpected. Please update trace collector."
                << std::endl;
  } break;
  }
}

void syclPrintersInit() {
  std::string_view PrinterType(std::getenv("SYCL_TRACE_PRINT_FORMAT"));
  if (PrinterType == "classic") {
    std::cerr << "Classic output is not supported yet for SYCL API\n";
  } else if (PrinterType == "verbose") {
    PrintSyclVerbose = true;
  } else if (PrinterType == "compact") {
    PrintSyclVerbose = false;
  }
}

// For unification purpose
void syclPrintersFinish() {}
