//==---------------------- pi_trace_collector.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi_trace_collector.cpp
/// Routines to collect and print Plugin Interface calls.

#include "xpti/xpti_trace_framework.h"

#include "pi_arguments_handler.hpp"
#include "pi_structs.hpp"

#include <CL/sycl/detail/spinlock.hpp>
#include <detail/plugin_printers.hpp>

#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

extern sycl::detail::SpinLock GlobalLock;

extern bool HasZEPrinter;
extern bool HasPIPrinter;

using HeaderPrinterT =
    std::function<void(const pi_plugin &, const xpti::function_with_args_t *)>;

static sycl::xpti_helpers::PiArgumentsHandler *ArgHandler = nullptr;
static HeaderPrinterT *HeaderPrinter = nullptr;
static std::function<void(pi_result)> *ResultPrinter = nullptr;

static void setupClassicPrinter() {
  ArgHandler = new sycl::xpti_helpers::PiArgumentsHandler();
#define _PI_API(api)                                                           \
  ArgHandler->set##_##api(                                                     \
      [](const pi_plugin &, std::optional<pi_result>, auto &&... Args) {       \
        std::cout << "---> " << #api << "("                                    \
                  << "\n";                                                     \
        sycl::detail::pi::printArgs(Args...);                                  \
      });
#include <CL/sycl/detail/pi.def>
#undef _PI_API

  ResultPrinter = new std::function(
      [](pi_result Res) { std::cout << ") ---> " << Res << std::endl; });
  HeaderPrinter = new std::function(
      [](const pi_plugin &Plugin, const xpti::function_with_args_t *Data) {
        ArgHandler->handle(Data->function_id, Plugin, std::nullopt,
                           Data->args_data);
      });
}

static void setupPrettyPrinter() {
  HeaderPrinter = new std::function(
      [](const pi_plugin &, const xpti::function_with_args_t *Data) {
        std::cout << "[PI] " << Data->function_name << "(\n";
        switch (Data->function_id) {
#include "pi_printers.def"
        }
        std::cout << ")";

        if (HasZEPrinter) {
          std::cout << " {" << std::endl;
        }
      });
  ResultPrinter = new std::function(
      [](pi_result Res) { std::cout << " ---> " << Res << std::endl; });
}

void piPrintersInit() {
  HasPIPrinter = true;
  std::string_view PrinterType(std::getenv("SYCL_TRACE_PI_PRINTER"));

  if (PrinterType == "classic") {
    setupClassicPrinter();
  } else if (PrinterType == "pretty") {
    setupPrettyPrinter();
  }
}

void piPrintersFinish() {
  if (ArgHandler)
    delete ArgHandler;
  delete HeaderPrinter;
  delete ResultPrinter;
}

XPTI_CALLBACK_API void piCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t * /*Parent*/,
                                  xpti::trace_event_data_t * /*Event*/,
                                  uint64_t /*Instance*/, const void *UserData) {
  if (!HeaderPrinter || !ResultPrinter)
    return;

  // Lock while we print information
  std::lock_guard _{GlobalLock};
  const auto *Data = static_cast<const xpti::function_with_args_t *>(UserData);
  if (TraceType == xpti::trace_function_with_args_begin) {
    const auto *Plugin = static_cast<pi_plugin *>(Data->user_data);
    (*HeaderPrinter)(*Plugin, Data);
  } else if (TraceType == xpti::trace_function_with_args_end) {
    (*ResultPrinter)(*static_cast<pi_result *>(Data->ret_data));
  }
}
