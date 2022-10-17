//==----------- cuda_trace_collector.cpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file cuda_trace_collector.cpp
/// Routines to collect and print CUDA Driver API calls.

#include "xpti/xpti_trace_framework.h"

#include <sycl/detail/spinlock.hpp>

#include <cuda.h>
#include <cupti.h>

#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

extern sycl::detail::SpinLock GlobalLock;

extern bool HasZEPrinter;
extern bool HasCUPrinter;
extern bool HasPIPrinter;

static bool PrintVerbose = false;

static std::string getResult(CUresult Res) {
  const char *Err;
  cuGetErrorName(Res, &Err);

  std::string ResultStr{Err};

  if (PrintVerbose) {
    const char *Desc;
    cuGetErrorString(Res, &Desc);
    ResultStr += " (" + std::string{Desc} + ")";
  }

  return ResultStr;
}

XPTI_CALLBACK_API void cuCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t * /*Parent*/,
                                  xpti::trace_event_data_t * /*Event*/,
                                  uint64_t /*Instance*/, const void *UserData) {
  std::lock_guard _{GlobalLock};
  const auto *Data = static_cast<const xpti::function_with_args_t *>(UserData);
  const auto PrintPrefix = [] {
    if (HasPIPrinter)
      std::cout << "*  ";
  };
  if (TraceType == xpti::trace_function_with_args_begin) {

    const auto PrintOffset = [PrintPrefix]() {
      PrintPrefix();
      std::cout << "   ";
    };

    PrintPrefix();
    if (PrintVerbose) {
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
      std::cout << "[CU:TID " << TID << ":";
      std::cout << Source << ":" << Line << "]\n";
      PrintPrefix();
    } else {
      std::cout << "[CU] ";
    }

    std::cout << Data->function_name << "(\n";

    switch (Data->function_id) {
#include "cuda_printers.def"
    default:
      break; // unknown API
    }

    if (HasPIPrinter) {
      std::cout << "*  ";
    }
    std::cout << std::flush;
  } else if (TraceType == xpti::trace_function_with_args_end) {
    std::cout << ") ---> "
              << getResult(*static_cast<CUresult *>(Data->ret_data))
              << std::endl;
    PrintPrefix();
    std::cout << std::endl;
  }
}

void cuPrintersInit() {
  HasCUPrinter = true;

  std::string_view PrinterType(std::getenv("SYCL_TRACE_PRINT_FORMAT"));
  if (PrinterType == "classic") {
    std::cerr << "Classic output is unsupported for CUDA\n";
  } else if (PrinterType == "verbose") {
    PrintVerbose = true;
  } else if (PrinterType == "compact") {
    PrintVerbose = false;
  }
}

// For unification purpose
void cuPrintersFinish() {}
