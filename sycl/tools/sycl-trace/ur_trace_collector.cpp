//==---------------------- ur_trace_collector.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file ur_trace_collector.cpp
/// Routines to collect and print Unified Runtime calls.

#include "xpti/xpti_trace_framework.h"

#include <sycl/detail/spinlock.hpp>
#include <ur_print.hpp>

#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

extern sycl::detail::SpinLock GlobalLock;

extern bool HasZEPrinter;

using PrinterT = std::function<void(const xpti::function_with_args_t *)>;

static PrinterT *HeaderPrinter = nullptr;
static PrinterT *ResultPrinter = nullptr;

static void setupClassicPrinter() {
  ResultPrinter = new std::function([](const xpti::function_with_args_t *Data) {
    ur::extras::printFunctionParams(
        std::cout, static_cast<ur_function_t>(Data->function_id),
        Data->args_data);
    auto *result = static_cast<const ur_result_t *>(Data->ret_data);

    std::cout << ")\n---> " << *result << "\n\n";
  });

  HeaderPrinter = new std::function([](const xpti::function_with_args_t *Data) {
    std::cout << "---> " << Data->function_name << "(\n";
  });
}

static void setupPrettyPrinter(bool Verbose) {
  HeaderPrinter =
      new std::function([Verbose](const xpti::function_with_args_t *Data) {
        if (Verbose) {
          std::string Source = "<unknown>";
          size_t Line = 0;

          auto *Payload = xptiQueryPayloadByUID(xptiGetUniversalId());

          if (Payload) {
            if (Payload->source_file != nullptr) {
              Source = Payload->source_file;
              Line = Payload->line_no;
            }
          }

          auto TID = std::this_thread::get_id();
          std::cout << "[UR:TID " << TID << ":";
          std::cout << Source << ":" << Line << "]\n";
        } else {
          std::cout << "[UR] ";
        }
        std::cout << Data->function_name << "(\n";

        if (HasZEPrinter) {
          std::cout << " {" << std::endl;
        }
      });

  ResultPrinter = new std::function([](const xpti::function_with_args_t *Data) {
    if (HasZEPrinter) {
      std::cout << "}";
    }
    std::cout << " ";
    ur::extras::printFunctionParams(
        std::cout, static_cast<ur_function_t>(Data->function_id),
        Data->args_data);
    auto *result = static_cast<const ur_result_t *>(Data->ret_data);

    std::cout << ")\n---> " << *result << "\n\n";
  });
}

void urPrintersInit() {
  std::string_view PrinterType(std::getenv("SYCL_TRACE_PRINT_FORMAT"));

  if (PrinterType == "classic") {
    setupClassicPrinter();
  } else if (PrinterType == "verbose") {
    setupPrettyPrinter(/*verbose*/ true);
  } else if (PrinterType == "compact") {
    setupPrettyPrinter(/*verbose*/ false);
  }
}

void urPrintersFinish() {
  delete HeaderPrinter;
  delete ResultPrinter;
}

XPTI_CALLBACK_API void urCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t * /*Parent*/,
                                  xpti::trace_event_data_t * /*Event*/,
                                  uint64_t /*Instance*/, const void *UserData) {
  if (!HeaderPrinter || !ResultPrinter)
    return;

  // Lock while we print information
  std::lock_guard<sycl::detail::SpinLock> _{GlobalLock};
  const auto *Data = static_cast<const xpti::function_with_args_t *>(UserData);
  if (TraceType == xpti::trace_function_with_args_begin) {
    (*HeaderPrinter)(Data);
  } else if (TraceType == xpti::trace_function_with_args_end) {
    (*ResultPrinter)(Data);
  }
}
