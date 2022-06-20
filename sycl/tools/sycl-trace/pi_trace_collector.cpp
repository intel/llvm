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

static std::string getResult(pi_result Res) {
  switch (Res) {
  case PI_SUCCESS:
    return "PI_SUCCESS";
  case PI_ERROR_INVALID_KERNEL_NAME:
    return "PI_ERROR_INVALID_KERNEL_NAME";
  case PI_ERROR_INVALID_OPERATION:
    return "CL_INVALID_OPERATION";
  case PI_ERROR_INVALID_KERNEL:
    return "PI_ERROR_INVALID_KERNEL";
  case PI_ERROR_INVALID_QUEUE_PROPERTIES:
    return "PI_ERROR_INVALID_QUEUE_PROPERTIES";
  case PI_ERROR_INVALID_VALUE:
    return "PI_ERROR_INVALID_VALUE";
  case PI_ERROR_INVALID_CONTEXT:
    return "PI_ERROR_INVALID_CONTEXT";
  case PI_ERROR_INVALID_PLATFORM:
    return "PI_ERROR_INVALID_PLATFORM";
  case PI_ERROR_INVALID_DEVICE:
    return "PI_ERROR_INVALID_DEVICE";
  case PI_ERROR_INVALID_BINARY:
    return "PI_ERROR_INVALID_BINARY";
  case PI_ERROR_INVALID_QUEUE:
    return "PI_INVALID_COMMAND_QUEUE";
  case PI_ERROR_OUT_OF_HOST_MEMORY:
    return "PI_ERROR_OUT_OF_HOST_MEMORY";
  case PI_ERROR_INVALID_PROGRAM:
    return "PI_ERROR_INVALID_PROGRAM";
  case PI_ERROR_INVALID_PROGRAM_EXECUTABLE:
    return "PI_ERROR_INVALID_PROGRAM_EXECUTABLE";
  case PI_ERROR_INVALID_SAMPLER:
    return "PI_ERROR_INVALID_SAMPLER";
  case PI_ERROR_INVALID_BUFFER_SIZE:
    return "PI_ERROR_INVALID_BUFFER_SIZE";
  case PI_ERROR_INVALID_MEM_OBJECT:
    return "PI_ERROR_INVALID_MEM_OBJECT";
  case PI_ERROR_OUT_OF_RESOURCES:
    return "PI_ERROR_OUT_OF_RESOURCES";
  case PI_ERROR_INVALID_EVENT:
    return "PI_ERROR_INVALID_EVENT";
  case PI_ERROR_INVALID_EVENT_WAIT_LIST:
    return "PI_ERROR_INVALID_EVENT_WAIT_LIST";
  case PI_ERROR_MISALIGNED_SUB_BUFFER_OFFSET:
    return "PI_ERROR_MISALIGNED_SUB_BUFFER_OFFSET";
  case PI_ERROR_BUILD_PROGRAM_FAILURE:
    return "PI_ERROR_BUILD_PROGRAM_FAILURE";
  case PI_ERROR_INVALID_WORK_GROUP_SIZE:
    return "PI_ERROR_INVALID_WORK_GROUP_SIZE";
  case PI_ERROR_COMPILER_NOT_AVAILABLE:
    return "PI_ERROR_COMPILER_NOT_AVAILABLE";
  case PI_ERROR_PROFILING_INFO_NOT_AVAILABLE:
    return "PI_ERROR_PROFILING_INFO_NOT_AVAILABLE";
  case PI_ERROR_DEVICE_NOT_FOUND:
    return "PI_ERROR_DEVICE_NOT_FOUND";
  case PI_ERROR_INVALID_WORK_ITEM_SIZE:
    return "PI_ERROR_INVALID_WORK_ITEM_SIZE";
  case PI_ERROR_INVALID_WORK_DIMENSION:
    return "PI_ERROR_INVALID_WORK_DIMENSION";
  case PI_ERROR_INVALID_KERNEL_ARGS:
    return "PI_ERROR_INVALID_KERNEL_ARGS";
  case PI_ERROR_INVALID_IMAGE_SIZE:
    return "PI_ERROR_INVALID_IMAGE_SIZE";
  case PI_ERROR_INVALID_ARG_VALUE:
    return "PI_ERROR_INVALID_ARG_VALUE";
  case PI_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "PI_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case PI_ERROR_IMAGE_FORMAT_NOT_SUPPORTED:
    return "PI_ERROR_IMAGE_FORMAT_NOT_SUPPORTED";
  case PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE:
    return "PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE";
  case PI_ERROR_LINK_PROGRAM_FAILURE:
    return "PI_ERROR_LINK_PROGRAM_FAILURE";
  case PI_ERROR_COMMAND_EXECUTION_FAILURE:
    return "PI_ERROR_COMMAND_EXECUTION_FAILURE";
  case PI_ERROR_FUNCTION_ADDRESS_IS_NOT_AVAILABLE:
    return "PI_ERROR_FUNCTION_ADDRESS_IS_NOT_AVAILABLE";
  case PI_ERROR_PLUGIN_SPECIFIC_ERROR:
    return "PI_ERROR_PLUGIN_SPECIFIC_ERROR";
  case PI_ERROR_UNKNOWN:
    return "PI_ERROR_UNKNOWN";
  }

  return "UNKNOWN RESULT";
}

static void setupClassicPrinter() {
  ArgHandler = new sycl::xpti_helpers::PiArgumentsHandler();
#define _PI_API(api)                                                           \
  ArgHandler->set##_##api(                                                     \
      [](const pi_plugin &, std::optional<pi_result>, auto &&...Args) {        \
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

static void setupPrettyPrinter(bool Verbose) {
  HeaderPrinter = new std::function(
      [Verbose](const pi_plugin &, const xpti::function_with_args_t *Data) {
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
          std::cout << "[PI:TID " << TID << ":";
          std::cout << Source << ":" << Line << "]\n";
        } else {
          std::cout << "[PI] ";
        }
        std::cout << Data->function_name << "(\n";
        switch (Data->function_id) {
#include "pi_printers.def"
        }
        std::cout << ")";

        if (HasZEPrinter) {
          std::cout << " {" << std::endl;
        }
      });
  ResultPrinter = new std::function([](pi_result Res) {
    if (HasZEPrinter) {
      std::cout << "}";
    }
    std::cout << " ---> " << getResult(Res) << "\n" << std::endl;
  });
}

void piPrintersInit() {
  HasPIPrinter = true;
  std::string_view PrinterType(std::getenv("SYCL_TRACE_PRINT_FORMAT"));

  if (PrinterType == "classic") {
    setupClassicPrinter();
  } else if (PrinterType == "verbose") {
    setupPrettyPrinter(/*verbose*/ true);
  } else if (PrinterType == "compact") {
    setupPrettyPrinter(/*verbose*/ false);
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
