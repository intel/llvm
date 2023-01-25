//==----------- sycl_trace_collector.cpp
//-------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file sycl_trace_collector.cpp
/// Routines to collect and print SYCL API calls.

#include "xpti/xpti_trace_framework.h"

#include <sycl/detail/spinlock.hpp>

#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

// contract with runtime:
enum XPTIPayloadType {
  UNDEFINED = 0,
  PAYLOAD_T,
  // FUNC_ARGS as example for future extension
};

struct XPTIUnifiedPayload {
  uint32_t Type;
  uint32_t Size;
  void *Data;
};

struct XPTIAggregatedData {
  uint32_t PayloadsCount;
  XPTIUnifiedPayload **Payloads;
};

extern sycl::detail::SpinLock GlobalLock;

extern bool HasSYCLPrinter;

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

XPTI_CALLBACK_API void syclCallback(uint16_t TraceType,
                                    xpti::trace_event_data_t *Parent,
                                    xpti::trace_event_data_t *Event,
                                    uint64_t /*Instance*/,
                                    const void *UserData) {
  std::lock_guard<sycl::detail::SpinLock> Lock{GlobalLock};
  if (TraceType == xpti::trace_diagnostics) {
    TraceDiagnosticsMessage(Parent, Event, static_cast<const char *>(UserData));
  } else if (PrintSyclVerbose)
    std::cout << "Trace type is unexpected. Please update trace collector."
              << std::endl;
}

void syclPrintersInit() {
  HasSYCLPrinter = true;

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