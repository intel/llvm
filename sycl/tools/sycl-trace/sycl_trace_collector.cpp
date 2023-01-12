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

extern sycl::detail::SpinLock GlobalLock;

extern bool HasSYPrinter;

bool PrintSyVerbose = false;

XPTI_CALLBACK_API void syCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t * /*Parent*/,
                                  xpti::trace_event_data_t * /*Event*/,
                                  uint64_t /*Instance*/, const void *UserData) {
  std::lock_guard<sycl::detail::SpinLock> _{GlobalLock};
  if (TraceType == xpti::trace_diagnostics) {
    std::cout << "[SYCL]" << static_cast<const char *>(UserData) << std::endl;
  } else
    std::cout << "TO REMOVE: seems to be error, unknown trace type "
              << std::endl;
}

void syPrintersInit() {
  HasSYPrinter = true;

  std::string_view PrinterType(std::getenv("SYCL_TRACE_PRINT_FORMAT"));
  if (PrinterType == "classic") {
    std::cerr << "Classic output is not supported yet for SYCL API\n";
  } else if (PrinterType == "verbose") {
    PrintSyVerbose = true;
  } else if (PrinterType == "compact") {
    PrintSyVerbose = false;
  }
}

// For unification purpose
void syPrintersFinish() {}