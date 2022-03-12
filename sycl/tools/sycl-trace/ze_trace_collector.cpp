//==----------- ze_trace_collector.cpp -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file ze_trace_collector.cpp
/// Routines to collect and print Level Zero API calls.

#include "xpti/xpti_trace_framework.h"

#include <CL/sycl/detail/spinlock.hpp>

#include <level_zero/zet_api.h>

#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

extern sycl::detail::SpinLock GlobalLock;

extern bool HasZEPrinter;
extern bool HasPIPrinter;

enum class ZEApiKind {
#define _ZE_API(call, domain, cb, params_type) call,
#include "../../plugins/level_zero/ze_api.def"
#undef _ZE_API
};

static std::string getResult(ze_result_t Res) { return ""; }

XPTI_CALLBACK_API void zeCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t * /*Parent*/,
                                  xpti::trace_event_data_t * /*Event*/,
                                  uint64_t /*Instance*/, const void *UserData) {
  std::lock_guard _{GlobalLock};
  const auto *Data = static_cast<const xpti::function_with_args_t *>(UserData);
  if (TraceType == xpti::trace_function_with_args_begin) {
    const auto PrintOffset = [] {
      if (HasPIPrinter)
        std::cout << "*  ";
      std::cout << "   ";
    };

    if (HasPIPrinter) {
      std::cout << "*  ";
    }

    std::cout << "[L0] " << Data->function_name << "(\n";

    switch (Data->function_id) {
#include "ze_printers.def"
    default:
      break; // unknown API
    }

    if (HasPIPrinter) {
      std::cout << "*  ";
    }
  } else if (TraceType == xpti::trace_function_with_args_end) {
    std::cout << ") ---> "
              << getResult(*static_cast<ze_result_t *>(Data->ret_data));
  }
}

void zePrintersInit() { HasZEPrinter = true; }

// For unification purpose
void zePrintersFinish() {}
